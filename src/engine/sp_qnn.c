/*
 * Shannon-Prime / Phase 2.3 — minimal QNN HTP runner shim implementation.
 *
 * Status: scaffold complete. dlopen + interface-fetching is real.
 * load_binary + execute have explicit TODOs marking where the QNN
 * sequence calls go — next-session work to fill those in against
 * the API documented in PHASE_2_3_DESIGN.md.
 *
 * The scaffold compiles + links cleanly via NDK aarch64-clang against
 * the QAIRT include tree and dlopen-resolves the runtime libs that
 * we already validated on-device in Phase 2.1+2.2. So when the
 * implementation gaps close, the link surface and ABI are already
 * proven.
 *
 * Copyright (C) 2026 Ray Daniels. AGPLv3.
 */
#include "sp_qnn.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

/* QNN headers from the QAIRT install. The build script's -I path
 * supplies $(QAIRT_ROOT)/include/QNN so these resolve. */
#include "QnnCommon.h"
#include "QnnInterface.h"
#include "QnnDevice.h"
#include "QnnContext.h"
#include "System/QnnSystemInterface.h"
#include "System/QnnSystemContext.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpPerfInfrastructure.h"

/* ------------------------------------------------------------------ */
/* Internal state                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    void *backend_dlh;             /* dlopen of libQnnHtp.so */
    void *system_dlh;              /* dlopen of libQnnSystem.so */
    QNN_INTERFACE_VER_TYPE qnn;    /* core QNN function table */
    QNN_SYSTEM_INTERFACE_VER_TYPE qsys; /* QnnSystem function table */
    int initialized;

    /* Shared backend + device + log across all sp_qnn_handles. The QNN
     * convention is one backend/device per process; multiple contexts
     * (one per .bin) attach to that single device. Earlier per-handle
     * allocation compounded HTP memory pressure (each split spent its
     * own backend+device state) and broke 2-split residency. Sharing
     * keeps the per-context cost to just the context+graph+sysCtx state. */
    Qnn_BackendHandle_t backend;
    Qnn_DeviceHandle_t  device;
    Qnn_LogHandle_t     log;
    uint32_t            power_cfg_id;     /* shared HTP burst mode cfg */
    int                 device_initialized;
} sp_qnn_lib_t;

static sp_qnn_lib_t g_lib = {0};

/* Per-handle persistent ION-backed tensor allocation tracking.
 * Phase 2.6b: when sp_qnn_alloc_persistent() registers a MemHandle for
 * a tensor input, we record the (rpcmem ptr, QNN handle) pair so
 * sp_qnn_destroy() can deregister + free in the right order. */
typedef struct {
    size_t          tensor_idx;     /* which input slot */
    void           *user_ptr;       /* rpcmem_alloc'd virtual address */
    Qnn_MemHandle_t mem_handle;     /* QnnMem_register'd handle */
    size_t          bytes;
} sp_qnn_persistent_t;

#define SP_QNN_MAX_PERSISTENT 8

struct sp_qnn_handle {
    Qnn_ContextHandle_t context;
    Qnn_GraphHandle_t   graph;

    /* Owned mmap'd context-binary bytes — kept alive for the context
     * lifetime since QnnContext_createFromBinary may reference it. */
    void  *bin_data;
    size_t bin_size;

    /* QnnSystemContext handle — must live for the lifetime of any
     * binary_info pointers we cached. Freed at destroy(). */
    QnnSystemContext_Handle_t sysCtx;

    /* Binary-info-derived metadata (lifetime-tied to sysCtx). */
    const QnnSystemContext_BinaryInfo_t *binary_info;

    /* Cached I/O tensor info for sp_qnn_get_io_info(). Allocated. */
    sp_qnn_tensor_info *inputs;
    size_t              n_inputs;
    sp_qnn_tensor_info *outputs;
    size_t              n_outputs;

    /* Tensor templates copied from binary_info at load — used at every
     * execute() call. We mutate clientBuf.{data,dataSize} per call. */
    Qnn_Tensor_t *in_tensors;    /* n_inputs entries */
    Qnn_Tensor_t *out_tensors;   /* n_outputs entries */

    /* Phase 2.6b: persistent ION-backed tensor allocations. Recorded by
     * sp_qnn_alloc_persistent(); freed by sp_qnn_destroy(). */
    sp_qnn_persistent_t persistent[SP_QNN_MAX_PERSISTENT];
    size_t              n_persistent;

    /* power_cfg_id moved to g_lib (shared across handles); per-handle
     * htp_disable_burst_mode is gone. */
};

/* ------------------------------------------------------------------ */
/* rpcmem dlopen — for Phase 2.6b persistent ION-backed tensor inputs */
/* ------------------------------------------------------------------ */

/* Heap IDs come from rpcmem.h in the Hexagon SDK. We hardcode the V69
 * value so sp_qnn.c stays self-contained — it doesn't have to pull in
 * the SDK headers (the FastRPC bridge in shannon_prime_hexagon.c does).
 * RPCMEM_HEAP_ID_SYSTEM = 25 = non-contiguous physical memory routed
 * through SMMU. RPCMEM_DEFAULT_FLAGS = 1 = ION_FLAG_CACHED. */
#ifndef SP_RPCMEM_HEAP_ID_SYSTEM
#define SP_RPCMEM_HEAP_ID_SYSTEM 25
#endif
#ifndef SP_RPCMEM_DEFAULT_FLAGS
#define SP_RPCMEM_DEFAULT_FLAGS 1
#endif

typedef void  (*sp_rpcmem_init_fn_t)(void);
typedef void  (*sp_rpcmem_deinit_fn_t)(void);
typedef void *(*sp_rpcmem_alloc_fn_t)(int heapid, uint32_t flags, size_t bytes);
typedef void  (*sp_rpcmem_free_fn_t)(void *po);
typedef int   (*sp_rpcmem_to_fd_fn_t)(void *po);

typedef struct {
    void *dlh;
    sp_rpcmem_init_fn_t   init;
    sp_rpcmem_deinit_fn_t deinit;
    sp_rpcmem_alloc_fn_t  alloc;
    sp_rpcmem_free_fn_t   free_;
    sp_rpcmem_to_fd_fn_t  to_fd;
    int initialized;
    int init_attempted;
} sp_qnn_rpcmem_t;

static sp_qnn_rpcmem_t g_rpc = {0};

static int sp_qnn_rpcmem_load(void) {
    if (g_rpc.initialized)      return 0;
    if (g_rpc.init_attempted)   return -1;
    g_rpc.init_attempted = 1;

    /* libcdsprpc.so is the consumer-facing FastRPC client lib that
     * exposes rpcmem_*. Both Android (S22U) and the Hexagon SDK
     * Linux samples use this name. */
    g_rpc.dlh = dlopen("libcdsprpc.so", RTLD_LAZY | RTLD_LOCAL);
    if (!g_rpc.dlh) {
        fprintf(stderr, "[sp_qnn] dlopen(libcdsprpc.so) failed: %s\n", dlerror());
        return -1;
    }
    g_rpc.init   = (sp_rpcmem_init_fn_t)  dlsym(g_rpc.dlh, "rpcmem_init");
    g_rpc.deinit = (sp_rpcmem_deinit_fn_t)dlsym(g_rpc.dlh, "rpcmem_deinit");
    g_rpc.alloc  = (sp_rpcmem_alloc_fn_t) dlsym(g_rpc.dlh, "rpcmem_alloc");
    g_rpc.free_  = (sp_rpcmem_free_fn_t)  dlsym(g_rpc.dlh, "rpcmem_free");
    g_rpc.to_fd  = (sp_rpcmem_to_fd_fn_t) dlsym(g_rpc.dlh, "rpcmem_to_fd");

    if (!g_rpc.alloc || !g_rpc.free_ || !g_rpc.to_fd) {
        fprintf(stderr, "[sp_qnn] rpcmem_alloc/free/to_fd missing from libcdsprpc.so\n");
        dlclose(g_rpc.dlh); g_rpc.dlh = NULL;
        return -1;
    }
    if (g_rpc.init) g_rpc.init();   /* refcounted in the SDK */
    g_rpc.initialized = 1;
    fprintf(stderr, "[sp_qnn] rpcmem loaded — ION-backed persistent tensors enabled\n");
    return 0;
}

/* ------------------------------------------------------------------ */
/* dlopen + interface-fetch plumbing — the part that's complete.      */
/* ------------------------------------------------------------------ */

static void *dl_load(const char *path) {
    void *h = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "[sp_qnn] dlopen(%s) failed: %s\n", path, dlerror());
    }
    return h;
}

/* Pull the latest-version interface from the providers array — the
 * QNN convention is "highest minor wins for a given major you support". */
static int pick_qnn_interface(QnnInterface_t *const *providers, uint32_t n,
                              QNN_INTERFACE_VER_TYPE *out) {
    /* Major version we're building against. QNN promises ABI stability
     * within a major version. */
    const uint32_t want_major = QNN_API_VERSION_MAJOR;
    int best = -1;
    uint32_t best_minor = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const Qnn_ApiVersion_t *v = &providers[i]->apiVersion;
        if (v->coreApiVersion.major != want_major) continue;
        if ((int)i == best && v->coreApiVersion.minor < best_minor) continue;
        best = (int)i;
        best_minor = v->coreApiVersion.minor;
    }
    if (best < 0) return -1;
    *out = providers[best]->QNN_INTERFACE_VER_NAME;
    return 0;
}

static int pick_qsys_interface(QnnSystemInterface_t *const *providers, uint32_t n,
                               QNN_SYSTEM_INTERFACE_VER_TYPE *out) {
    const uint32_t want_major = QNN_SYSTEM_API_VERSION_MAJOR;
    int best = -1;
    uint32_t best_minor = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const Qnn_Version_t *v = &providers[i]->systemApiVersion;
        if (v->major != want_major) continue;
        if ((int)i == best && v->minor < best_minor) continue;
        best = (int)i;
        best_minor = v->minor;
    }
    if (best < 0) return -1;
    *out = providers[best]->QNN_SYSTEM_INTERFACE_VER_NAME;
    return 0;
}

sp_qnn_status sp_qnn_init(const char *backend_lib_path,
                          const char *system_lib_path) {
    if (g_lib.initialized) return SP_QNN_OK;

    if (!backend_lib_path) backend_lib_path = "libQnnHtp.so";
    if (!system_lib_path)  system_lib_path  = "libQnnSystem.so";

    /* 1. Open backend (HTP). */
    g_lib.backend_dlh = dl_load(backend_lib_path);
    if (!g_lib.backend_dlh) return SP_QNN_ERR_DLOPEN;

    typedef Qnn_ErrorHandle_t (*QnnInterface_getProviders_t)(
        const QnnInterface_t ***, uint32_t *);
    QnnInterface_getProviders_t getProviders =
        (QnnInterface_getProviders_t)dlsym(g_lib.backend_dlh,
                                           "QnnInterface_getProviders");
    if (!getProviders) {
        fprintf(stderr, "[sp_qnn] dlsym(QnnInterface_getProviders) failed: %s\n", dlerror());
        return SP_QNN_ERR_GET_INTERFACE;
    }
    const QnnInterface_t **providers = NULL;
    uint32_t n = 0;
    if (getProviders(&providers, &n) != QNN_SUCCESS || n == 0) {
        return SP_QNN_ERR_GET_INTERFACE;
    }
    if (pick_qnn_interface(providers, n, &g_lib.qnn) != 0) {
        fprintf(stderr, "[sp_qnn] no compatible QNN interface (want major %u)\n",
                QNN_API_VERSION_MAJOR);
        return SP_QNN_ERR_GET_INTERFACE;
    }

    /* 2. Open System lib (for context-binary info parsing). */
    g_lib.system_dlh = dl_load(system_lib_path);
    if (!g_lib.system_dlh) return SP_QNN_ERR_DLOPEN;

    typedef Qnn_ErrorHandle_t (*QnnSystemInterface_getProviders_t)(
        const QnnSystemInterface_t ***, uint32_t *);
    QnnSystemInterface_getProviders_t sysGetProviders =
        (QnnSystemInterface_getProviders_t)dlsym(g_lib.system_dlh,
                                                  "QnnSystemInterface_getProviders");
    if (!sysGetProviders) {
        fprintf(stderr, "[sp_qnn] dlsym(QnnSystemInterface_getProviders) failed: %s\n",
                dlerror());
        return SP_QNN_ERR_GET_INTERFACE;
    }
    const QnnSystemInterface_t **sysProviders = NULL;
    uint32_t sysN = 0;
    if (sysGetProviders(&sysProviders, &sysN) != QNN_SUCCESS || sysN == 0) {
        return SP_QNN_ERR_GET_INTERFACE;
    }
    if (pick_qsys_interface(sysProviders, sysN, &g_lib.qsys) != 0) {
        return SP_QNN_ERR_GET_INTERFACE;
    }

    g_lib.initialized = 1;
    fprintf(stderr, "[sp_qnn] initialized: backend=%s system=%s\n",
            backend_lib_path, system_lib_path);
    return SP_QNN_OK;
}

/* Forward decls so we can call burst-mode helpers from init_device. */
static void htp_enable_burst_mode_shared(void);
static void htp_disable_burst_mode_shared(void);

/* Lazily create the shared backend + device + log on first load_binary.
 * Sharing these across all sp_qnn_handles avoids the per-split state
 * compounding that breaks 2-split residency on V69 HTP. */
static sp_qnn_status sp_qnn_init_device(void) {
    if (g_lib.device_initialized) return SP_QNN_OK;

    if (g_lib.qnn.logCreate(NULL, QNN_LOG_LEVEL_WARN, &g_lib.log) != QNN_SUCCESS) {
        g_lib.log = NULL;  /* non-fatal */
    }

    if (g_lib.qnn.backendCreate(g_lib.log, NULL, &g_lib.backend) != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] shared backendCreate failed\n");
        if (g_lib.log) g_lib.qnn.logFree(g_lib.log);
        return SP_QNN_ERR_BACKEND_CREATE;
    }

    if (g_lib.qnn.deviceCreate(g_lib.log, NULL, &g_lib.device) != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] shared deviceCreate failed\n");
        g_lib.qnn.backendFree(g_lib.backend);
        if (g_lib.log) g_lib.qnn.logFree(g_lib.log);
        return SP_QNN_ERR_DEVICE_CREATE;
    }

    htp_enable_burst_mode_shared();
    g_lib.device_initialized = 1;
    fprintf(stderr, "[sp_qnn] shared backend+device ready (one-time)\n");
    return SP_QNN_OK;
}

void sp_qnn_shutdown(void) {
    if (g_lib.device_initialized) {
        htp_disable_burst_mode_shared();
        if (g_lib.device)  g_lib.qnn.deviceFree(g_lib.device);
        if (g_lib.backend) g_lib.qnn.backendFree(g_lib.backend);
        if (g_lib.log)     g_lib.qnn.logFree(g_lib.log);
        g_lib.device  = NULL;
        g_lib.backend = NULL;
        g_lib.log     = NULL;
        g_lib.device_initialized = 0;
    }
    if (g_lib.system_dlh)  { dlclose(g_lib.system_dlh);  g_lib.system_dlh = NULL; }
    if (g_lib.backend_dlh) { dlclose(g_lib.backend_dlh); g_lib.backend_dlh = NULL; }
    g_lib.initialized = 0;
}

/* ------------------------------------------------------------------ */
/* File-load helpers                                                  */
/* ------------------------------------------------------------------ */

/* mmap the .bin instead of read+malloc. Per the original Phase 2.4
 * architecture note ("will need to mmap-load contexts rather than
 * dlopen-into-RAM") this keeps the bin bytes file-backed so they
 * don't compete with HTP-side context state for our process RSS.
 *
 * MAP_PRIVATE: HTP can't dirty our pages.
 * MAP_POPULATE on Android (when supported) prefaults so the first
 *   contextCreateFromBinary doesn't pay page-fault cost mid-call.
 */
static int read_file(const char *path, void **out_data, size_t *out_size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }
    int flags = MAP_PRIVATE;
#ifdef MAP_POPULATE
    flags |= MAP_POPULATE;
#endif
    void *buf = mmap(NULL, (size_t)st.st_size, PROT_READ, flags, fd, 0);
    close(fd);  /* mmap holds its own ref */
    if (buf == MAP_FAILED) return -1;
    *out_data = buf;
    *out_size = (size_t)st.st_size;
    return 0;
}

/* Companion to read_file's mmap. Replaces free(h->bin_data). */
static void unmap_file(void *data, size_t size) {
    if (data && size > 0) munmap(data, size);
}

/* ------------------------------------------------------------------ */
/* Load + execute — TODO sections marked, see PHASE_2_3_DESIGN.md     */
/* ------------------------------------------------------------------ */

/* Resolve the per-graph Qnn_Tensor_t arrays from the versioned
 * binary_info struct. Returns 0 on success, -1 on unsupported version. */
static int resolve_graph_info(const QnnSystemContext_BinaryInfo_t *bi,
                              const char **out_graph_name,
                              const Qnn_Tensor_t **out_inputs,
                              uint32_t *out_n_in,
                              const Qnn_Tensor_t **out_outputs,
                              uint32_t *out_n_out) {
    /* binary_info is versioned. We pick graph 0 in either layout. */
    const QnnSystemContext_GraphInfo_t *graph0 = NULL;
    if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        if (bi->contextBinaryInfoV1.numGraphs == 0) return -1;
        graph0 = &bi->contextBinaryInfoV1.graphs[0];
    } else if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        if (bi->contextBinaryInfoV2.numGraphs == 0) return -1;
        graph0 = &bi->contextBinaryInfoV2.graphs[0];
    } else if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        if (bi->contextBinaryInfoV3.numGraphs == 0) return -1;
        graph0 = &bi->contextBinaryInfoV3.graphs[0];
    } else {
        fprintf(stderr, "[sp_qnn] unknown binary_info version: %d\n", (int)bi->version);
        return -1;
    }
    /* Per-graph info also versioned. */
    if (graph0->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        *out_graph_name = graph0->graphInfoV1.graphName;
        *out_inputs     = graph0->graphInfoV1.graphInputs;
        *out_n_in       = graph0->graphInfoV1.numGraphInputs;
        *out_outputs    = graph0->graphInfoV1.graphOutputs;
        *out_n_out      = graph0->graphInfoV1.numGraphOutputs;
    } else if (graph0->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        *out_graph_name = graph0->graphInfoV2.graphName;
        *out_inputs     = graph0->graphInfoV2.graphInputs;
        *out_n_in       = graph0->graphInfoV2.numGraphInputs;
        *out_outputs    = graph0->graphInfoV2.graphOutputs;
        *out_n_out      = graph0->graphInfoV2.numGraphOutputs;
    } else if (graph0->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        *out_graph_name = graph0->graphInfoV3.graphName;
        *out_inputs     = graph0->graphInfoV3.graphInputs;
        *out_n_in       = graph0->graphInfoV3.numGraphInputs;
        *out_outputs    = graph0->graphInfoV3.graphOutputs;
        *out_n_out      = graph0->graphInfoV3.numGraphOutputs;
    } else {
        return -1;
    }
    return 0;
}

/* Convert a Qnn_DataType_t to bytes per element (rough; QNN has more types). */
static uint32_t dtype_bytes(Qnn_DataType_t dt) {
    switch (dt) {
    case QNN_DATATYPE_FLOAT_32: case QNN_DATATYPE_INT_32: case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_SFIXED_POINT_32: case QNN_DATATYPE_UFIXED_POINT_32:
        return 4;
    case QNN_DATATYPE_FLOAT_16: case QNN_DATATYPE_INT_16: case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_SFIXED_POINT_16: case QNN_DATATYPE_UFIXED_POINT_16:
        return 2;
    case QNN_DATATYPE_INT_8: case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8: case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_BOOL_8:
        return 1;
    case QNN_DATATYPE_FLOAT_64: case QNN_DATATYPE_INT_64: case QNN_DATATYPE_UINT_64:
        return 8;
    default:
        return 0;
    }
}

/* Cache one tensor template into our flat sp_qnn_tensor_info struct. */
static void cache_tensor_info(const Qnn_Tensor_t *t, sp_qnn_tensor_info *out) {
    if (t->version == QNN_TENSOR_VERSION_1) {
        out->name              = t->v1.name;
        out->rank              = t->v1.rank;
        out->dims              = t->v1.dimensions;
        out->dtype             = (uint32_t)t->v1.dataType;
        out->bytes_per_element = dtype_bytes(t->v1.dataType);
    } else {
        out->name              = t->v2.name;
        out->rank              = t->v2.rank;
        out->dims              = t->v2.dimensions;
        out->dtype             = (uint32_t)t->v2.dataType;
        out->bytes_per_element = dtype_bytes(t->v2.dataType);
    }
}

/* Enable HTP burst mode (DCVS off, perf governor pinned high) for the
 * given device. Match qnn-net-run's defaults. Sets h->power_cfg_id on
 * success so destroy can clean up. Best-effort: any failure is logged
 * but doesn't fail the whole load (we can still execute, just slower). */
static void htp_enable_burst_mode_shared(void) {
    QnnDevice_Infrastructure_t infra_raw = NULL;
    if (g_lib.qnn.deviceGetInfrastructure(&infra_raw) != QNN_SUCCESS || !infra_raw) {
        fprintf(stderr, "[sp_qnn] perf-mode: deviceGetInfrastructure failed (skipping burst)\n");
        return;
    }
    QnnHtpDevice_Infrastructure_t *infra = (QnnHtpDevice_Infrastructure_t *)infra_raw;
    if (infra->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
        fprintf(stderr, "[sp_qnn] perf-mode: not an HTP perf-infra (type=%d)\n",
                (int)infra->infraType);
        return;
    }
    QnnHtpDevice_PerfInfrastructure_t *perf = &infra->perfInfra;

    uint32_t cfg_id = 0;
    if (perf->createPowerConfigId(0, 0, &cfg_id) != QNN_SUCCESS || cfg_id == 0) {
        fprintf(stderr, "[sp_qnn] perf-mode: createPowerConfigId failed\n");
        return;
    }

    QnnHtpPerfInfrastructure_PowerConfig_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    cfg.dcvsV3Config.contextId          = cfg_id;
    cfg.dcvsV3Config.setDcvsEnable      = 1;
    cfg.dcvsV3Config.dcvsEnable         = 0;
    cfg.dcvsV3Config.powerMode          = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    cfg.dcvsV3Config.setSleepLatency    = 1;
    cfg.dcvsV3Config.sleepLatency       = 40;
    cfg.dcvsV3Config.setBusParams       = 1;
    cfg.dcvsV3Config.busVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_TURBO;
    cfg.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
    cfg.dcvsV3Config.busVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_TURBO;
    cfg.dcvsV3Config.setCoreParams      = 1;
    cfg.dcvsV3Config.coreVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_TURBO;
    cfg.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
    cfg.dcvsV3Config.coreVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_TURBO;

    const QnnHtpPerfInfrastructure_PowerConfig_t *cfgs[] = { &cfg, NULL };
    if (perf->setPowerConfig(cfg_id, cfgs) != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] perf-mode: setPowerConfig failed\n");
        perf->destroyPowerConfigId(cfg_id);
        return;
    }
    g_lib.power_cfg_id = cfg_id;
    fprintf(stderr, "[sp_qnn] HTP burst mode enabled (shared cfg_id=%u)\n", cfg_id);
}

static void htp_disable_burst_mode_shared(void) {
    if (g_lib.power_cfg_id == 0) return;
    QnnDevice_Infrastructure_t infra_raw = NULL;
    if (g_lib.qnn.deviceGetInfrastructure(&infra_raw) == QNN_SUCCESS && infra_raw) {
        QnnHtpDevice_Infrastructure_t *infra =
            (QnnHtpDevice_Infrastructure_t *)infra_raw;
        if (infra->infraType == QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
            infra->perfInfra.destroyPowerConfigId(g_lib.power_cfg_id);
        }
    }
    g_lib.power_cfg_id = 0;
}

sp_qnn_status sp_qnn_load_binary(const char *context_bin_path,
                                 const char *graph_name,
                                 sp_qnn_handle **out_h) {
    if (!g_lib.initialized || !context_bin_path || !out_h)
        return SP_QNN_ERR_INVALID;

    sp_qnn_handle *h = (sp_qnn_handle *)calloc(1, sizeof(*h));
    if (!h) return SP_QNN_ERR_INVALID;

    /* (1) Read context binary into memory. */
    if (read_file(context_bin_path, &h->bin_data, &h->bin_size) != 0) {
        fprintf(stderr, "[sp_qnn] read_file(%s) failed\n", context_bin_path);
        free(h);
        return SP_QNN_ERR_READ_FILE;
    }
    fprintf(stderr, "[sp_qnn] loaded %zu bytes from %s\n",
            h->bin_size, context_bin_path);

    /* (2) QnnSystemContext to extract binary info (graphs, tensors). */
    if (g_lib.qsys.systemContextCreate(&h->sysCtx) != QNN_SUCCESS) {
        unmap_file(h->bin_data, h->bin_size); free(h);
        return SP_QNN_ERR_BINARY_INFO;
    }
    Qnn_ContextBinarySize_t info_size = 0;
    if (g_lib.qsys.systemContextGetBinaryInfo(
            h->sysCtx, h->bin_data, h->bin_size,
            &h->binary_info, &info_size) != QNN_SUCCESS) {
        g_lib.qsys.systemContextFree(h->sysCtx);
        unmap_file(h->bin_data, h->bin_size); free(h);
        return SP_QNN_ERR_BINARY_INFO;
    }

    /* (3) Resolve the graph + tensor templates from binary_info. */
    const char *bin_graph_name = NULL;
    const Qnn_Tensor_t *bin_inputs = NULL, *bin_outputs = NULL;
    uint32_t n_in = 0, n_out = 0;
    if (resolve_graph_info(h->binary_info, &bin_graph_name,
                           &bin_inputs, &n_in,
                           &bin_outputs, &n_out) != 0) {
        fprintf(stderr, "[sp_qnn] unsupported binary_info version\n");
        g_lib.qsys.systemContextFree(h->sysCtx);
        unmap_file(h->bin_data, h->bin_size); free(h);
        return SP_QNN_ERR_BINARY_INFO;
    }
    fprintf(stderr, "[sp_qnn] graph '%s': %u inputs / %u outputs\n",
            bin_graph_name ? bin_graph_name : "(unnamed)", n_in, n_out);

    /* (4) Lazily create the SHARED backend + device + log + burst mode.
     *     Subsequent loads reuse them — no per-handle backend/device
     *     allocation, which broke 2-split residency on V69. */
    if (sp_qnn_init_device() != SP_QNN_OK) {
        g_lib.qsys.systemContextFree(h->sysCtx);
        unmap_file(h->bin_data, h->bin_size); free(h);
        return SP_QNN_ERR_BACKEND_CREATE;
    }

    /* (5) Create context FROM the binary with HIGH priority hint, attached
     *     to the SHARED backend + device. */
    QnnContext_Config_t prio_cfg;
    memset(&prio_cfg, 0, sizeof(prio_cfg));
    prio_cfg.option   = QNN_CONTEXT_CONFIG_OPTION_PRIORITY;
    prio_cfg.priority = QNN_PRIORITY_HIGH;
    const QnnContext_Config_t *ctx_cfgs[] = { &prio_cfg, NULL };

    if (g_lib.qnn.contextCreateFromBinary(g_lib.backend, g_lib.device,
                                          ctx_cfgs,
                                          h->bin_data, h->bin_size,
                                          &h->context,
                                          NULL /*profile*/) != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] contextCreateFromBinary failed\n");
        g_lib.qsys.systemContextFree(h->sysCtx);
        unmap_file(h->bin_data, h->bin_size); free(h);
        return SP_QNN_ERR_CONTEXT_CREATE;
    }

    /* (6) Retrieve the named graph (or first). */
    const char *gname_use = graph_name ? graph_name : bin_graph_name;
    if (g_lib.qnn.graphRetrieve(h->context, gname_use, &h->graph) != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] graphRetrieve('%s') failed\n", gname_use);
        g_lib.qnn.contextFree(h->context, NULL);
        g_lib.qsys.systemContextFree(h->sysCtx);
        unmap_file(h->bin_data, h->bin_size); free(h);
        return SP_QNN_ERR_GRAPH_RETRIEVE;
    }

    /* (9) Cache I/O tensor templates and metadata. We allocate three
     *     parallel arrays per side: the public sp_qnn_tensor_info, and
     *     the internal Qnn_Tensor_t templates (copied from binary_info
     *     so the originals can be freed when sysCtx is freed). */
    h->n_inputs   = n_in;
    h->n_outputs  = n_out;
    h->inputs     = (sp_qnn_tensor_info *)calloc(n_in,  sizeof(*h->inputs));
    h->outputs    = (sp_qnn_tensor_info *)calloc(n_out, sizeof(*h->outputs));
    h->in_tensors = (Qnn_Tensor_t *)calloc(n_in,  sizeof(*h->in_tensors));
    h->out_tensors= (Qnn_Tensor_t *)calloc(n_out, sizeof(*h->out_tensors));
    if ((n_in > 0 && (!h->inputs || !h->in_tensors)) ||
        (n_out > 0 && (!h->outputs || !h->out_tensors))) {
        sp_qnn_destroy(&h);
        return SP_QNN_ERR_INVALID;
    }
    for (uint32_t i = 0; i < n_in;  ++i) {
        h->in_tensors[i]  = bin_inputs[i];   /* shallow copy template */
        cache_tensor_info(&bin_inputs[i],  &h->inputs[i]);
    }
    for (uint32_t i = 0; i < n_out; ++i) {
        h->out_tensors[i] = bin_outputs[i];
        cache_tensor_info(&bin_outputs[i], &h->outputs[i]);
    }

    fprintf(stderr, "[sp_qnn] context + graph ready\n");
    *out_h = h;
    return SP_QNN_OK;
}

void sp_qnn_destroy(sp_qnn_handle **h_io) {
    if (!h_io || !*h_io) return;
    sp_qnn_handle *h = *h_io;

    /* Phase 2.6b: deregister persistent ION-backed tensors before
     * destroying the context. Order matters — handles reference
     * the context, ION fds reference the handles. */
    if (h->n_persistent && g_lib.qnn.memDeRegister) {
        for (size_t i = 0; i < h->n_persistent; ++i) {
            sp_qnn_persistent_t *p = &h->persistent[i];
            if (p->mem_handle) {
                g_lib.qnn.memDeRegister(&p->mem_handle, 1);
            }
            if (p->user_ptr && g_rpc.free_) {
                g_rpc.free_(p->user_ptr);
            }
        }
        h->n_persistent = 0;
    }

    /* Release per-handle resources only. Shared backend/device/log/burst
     * persist across handles and are released in sp_qnn_shutdown(). */
    if (h->context) g_lib.qnn.contextFree(h->context, NULL);
    if (h->sysCtx)  g_lib.qsys.systemContextFree(h->sysCtx);

    free(h->in_tensors);
    free(h->out_tensors);
    free(h->inputs);
    free(h->outputs);
    unmap_file(h->bin_data, h->bin_size);
    free(h);
    *h_io = NULL;
}

sp_qnn_status sp_qnn_get_io_info(sp_qnn_handle *h,
                                 size_t *out_n_inputs,
                                 const sp_qnn_tensor_info **out_inputs,
                                 size_t *out_n_outputs,
                                 const sp_qnn_tensor_info **out_outputs) {
    if (!h) return SP_QNN_ERR_INVALID;
    /* TODO populate from binary_info during load. */
    if (out_n_inputs)  *out_n_inputs  = h->n_inputs;
    if (out_inputs)    *out_inputs    = h->inputs;
    if (out_n_outputs) *out_n_outputs = h->n_outputs;
    if (out_outputs)   *out_outputs   = h->outputs;
    return SP_QNN_OK;
}

/* Set the (data, dataSize) on a tensor that's a copy of a binary_info
 * template. Handles version 1 vs 2 layouts.
 *
 * Skips the rebind if the tensor is already memhandle-backed (i.e., the
 * caller registered it via sp_qnn_register_persistent_input) OR if `data`
 * is NULL (which signals "use whatever was bound before"). */
static void tensor_set_buf(Qnn_Tensor_t *t, void *data, size_t bytes) {
    if (t->version == QNN_TENSOR_VERSION_1) {
        if (t->v1.memType == QNN_TENSORMEMTYPE_MEMHANDLE) return;
        if (!data) return;  /* keep prior binding */
        t->v1.memType            = QNN_TENSORMEMTYPE_RAW;
        t->v1.clientBuf.data     = data;
        t->v1.clientBuf.dataSize = (uint32_t)bytes;
    } else {
        if (t->v2.memType == QNN_TENSORMEMTYPE_MEMHANDLE) return;
        if (!data) return;
        t->v2.memType            = QNN_TENSORMEMTYPE_RAW;
        t->v2.clientBuf.data     = data;
        t->v2.clientBuf.dataSize = (uint32_t)bytes;
    }
}

sp_qnn_status sp_qnn_execute(sp_qnn_handle *h,
                             const void *const *inputs,
                             const size_t *input_sizes,
                             void *const *outputs,
                             const size_t *output_sizes,
                             uint64_t *out_exec_us) {
    if (!h) return SP_QNN_ERR_INVALID;
    if (!inputs && h->n_inputs)  return SP_QNN_ERR_INVALID;
    if (!outputs && h->n_outputs) return SP_QNN_ERR_INVALID;

    /* Bind the supplied buffers into our cached templates. The rest of
     * the tensor metadata (name, id, dataType, dims, type APP_WRITE/READ)
     * came from binary_info via the shallow copy at load time, so we
     * only need to update the data pointer + size per call. */
    for (size_t i = 0; i < h->n_inputs; ++i) {
        tensor_set_buf(&h->in_tensors[i],  (void *)inputs[i],  input_sizes[i]);
    }
    for (size_t i = 0; i < h->n_outputs; ++i) {
        tensor_set_buf(&h->out_tensors[i], outputs[i],         output_sizes[i]);
    }

    /* Wall-clock the QNN call. */
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    Qnn_ErrorHandle_t rc = g_lib.qnn.graphExecute(
        h->graph,
        h->in_tensors,  (uint32_t)h->n_inputs,
        h->out_tensors, (uint32_t)h->n_outputs,
        NULL /*profile*/, NULL /*signal*/);
    gettimeofday(&t1, NULL);

    if (out_exec_us) {
        *out_exec_us = (uint64_t)((t1.tv_sec  - t0.tv_sec)  * 1000000) +
                       (uint64_t)((t1.tv_usec - t0.tv_usec));
    }

    if (rc != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] graphExecute failed: 0x%lx\n", (unsigned long)rc);
        return SP_QNN_ERR_EXECUTE;
    }
    return SP_QNN_OK;
}

sp_qnn_status sp_qnn_bench(sp_qnn_handle *h,
                           const void *const *inputs,
                           const size_t *input_sizes,
                           void *const *outputs,
                           const size_t *output_sizes,
                           uint32_t n_iter,
                           sp_qnn_bench_result *out) {
    if (!h || !out || n_iter == 0) return SP_QNN_ERR_INVALID;
    out->n_iterations = 0;
    out->min_us = UINT64_MAX;
    out->avg_us = 0;
    out->max_us = 0;
    uint64_t sum = 0;
    for (uint32_t i = 0; i < n_iter; ++i) {
        uint64_t us = 0;
        sp_qnn_status rc = sp_qnn_execute(h, inputs, input_sizes,
                                          outputs, output_sizes, &us);
        if (rc != SP_QNN_OK) return rc;
        if (us < out->min_us) out->min_us = us;
        if (us > out->max_us) out->max_us = us;
        sum += us;
        out->n_iterations++;
    }
    out->avg_us = sum / n_iter;
    return SP_QNN_OK;
}

/* ------------------------------------------------------------------ */
/* List-load: createFromBinaryListAsync with shared HTP resources.    */
/* This is the canonical multi-context API that lets N .bins coexist  */
/* in HTP working memory by sharing kernel/workspace state between    */
/* contexts in the group.                                             */
/* ------------------------------------------------------------------ */

#include <pthread.h>

/* Per-context async-load state, threaded through QnnContext_Params_t.notifyParam.
 * The async API fires notifyFunc when each context's deserialization completes;
 * we record success/failure here for the join. */
typedef struct {
    sp_qnn_handle             *h;
    Qnn_ContextHandle_t        ctx;        /* set by notifyFunc on success */
    Qnn_GraphHandle_t          graph;      /* set by notifyFunc */
    const char                *graph_name; /* set by notifyFunc */
    Qnn_ErrorHandle_t          status;     /* QNN_SUCCESS or error code */
    int                        done;
    pthread_mutex_t           *mu;
    pthread_cond_t            *cv;
    int                       *remaining;  /* group counter */
} sp_qnn_async_slot_t;

static void sp_qnn_list_notify(Qnn_ContextHandle_t context,
                               Qnn_GraphHandle_t graph,
                               const char *graphName,
                               QnnContext_createFromBinaryAsyncNotifyType_t notifyType,
                               void *notifyParam,
                               Qnn_ErrorHandle_t status) {
    (void)notifyType;
    sp_qnn_async_slot_t *s = (sp_qnn_async_slot_t *)notifyParam;
    pthread_mutex_lock(s->mu);
    s->ctx        = context;
    s->graph      = graph;
    s->graph_name = graphName;
    s->status     = status;
    s->done       = 1;
    (*s->remaining)--;
    pthread_cond_broadcast(s->cv);
    pthread_mutex_unlock(s->mu);
}

sp_qnn_status sp_qnn_load_binary_list(const char *const *paths,
                                      const char *const *graph_names,
                                      size_t n,
                                      sp_qnn_handle **out_handles) {
    if (!g_lib.initialized || !paths || n == 0 || !out_handles)
        return SP_QNN_ERR_INVALID;

    /* Lazy-init shared backend+device. */
    if (sp_qnn_init_device() != SP_QNN_OK) return SP_QNN_ERR_BACKEND_CREATE;

    /* Allocate per-binary handles + system contexts up front so notifyFunc
     * can populate them. */
    sp_qnn_handle      **handles = calloc(n, sizeof(*handles));
    sp_qnn_async_slot_t *slots   = calloc(n, sizeof(*slots));
    QnnContext_Params_t *params  = calloc(n, sizeof(*params));
    const QnnContext_Params_t **paramPtrs = calloc(n + 1, sizeof(*paramPtrs));
    if (!handles || !slots || !params || !paramPtrs) {
        free(handles); free(slots); free(params); free(paramPtrs);
        return SP_QNN_ERR_INVALID;
    }

    /* Per-context config: PRIORITY_HIGH (same as single-load path). */
    QnnContext_Config_t prio_cfg;
    memset(&prio_cfg, 0, sizeof(prio_cfg));
    prio_cfg.option   = QNN_CONTEXT_CONFIG_OPTION_PRIORITY;
    prio_cfg.priority = QNN_PRIORITY_HIGH;
    static const QnnContext_Config_t *per_ctx_cfgs[] = { NULL, NULL };
    per_ctx_cfgs[0] = &prio_cfg;

    /* Group-level listConfig: HTP shareResources=true. This is the option
     * that's only honored under createFromBinaryListAsync — it lets the
     * contexts share underlying kernel/workspace state, which is the fix
     * for our case where 4 .bins exceeded HTP working memory individually.
     *
     * SP_QNN_NO_SHARE_RESOURCES env var = bypass this for diagnostic
     * (some V69 SDK builds don't honor the option and return NOT_SUPPORTED). */
    QnnHtpContext_CustomConfig_t htp_share = {
        .option = QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES,
        .shareResources = true,
    };
    QnnContext_Config_t list_cfg_share;
    memset(&list_cfg_share, 0, sizeof(list_cfg_share));
    list_cfg_share.option       = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    list_cfg_share.customConfig = &htp_share;
    static const QnnContext_Config_t *list_cfg_ptrs[] = { NULL, NULL };
    const QnnContext_Config_t **list_cfg = NULL;  /* NULL = no group config */
    if (!getenv("SP_QNN_NO_SHARE_RESOURCES")) {
        list_cfg_ptrs[0] = &list_cfg_share;
        list_cfg = list_cfg_ptrs;
        fprintf(stderr, "[sp_qnn] list-load using shareResources=true\n");
    } else {
        fprintf(stderr, "[sp_qnn] list-load WITHOUT shareResources (env override)\n");
    }

    /* Async coordination. */
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t  cv = PTHREAD_COND_INITIALIZER;
    int remaining = (int)n;

    /* mmap each .bin, alloc handle + sysCtx, build params struct. */
    for (size_t i = 0; i < n; ++i) {
        handles[i] = calloc(1, sizeof(*handles[i]));
        if (!handles[i]) goto cleanup_fail;

        if (read_file(paths[i], &handles[i]->bin_data, &handles[i]->bin_size) != 0) {
            fprintf(stderr, "[sp_qnn] mmap %s failed\n", paths[i]);
            goto cleanup_fail;
        }
        fprintf(stderr, "[sp_qnn] mmap'd %zu bytes from %s\n",
                handles[i]->bin_size, paths[i]);

        if (g_lib.qsys.systemContextCreate(&handles[i]->sysCtx) != QNN_SUCCESS) {
            fprintf(stderr, "[sp_qnn] systemContextCreate failed for %s\n", paths[i]);
            goto cleanup_fail;
        }
        Qnn_ContextBinarySize_t info_size = 0;
        if (g_lib.qsys.systemContextGetBinaryInfo(handles[i]->sysCtx,
                                                  handles[i]->bin_data,
                                                  handles[i]->bin_size,
                                                  &handles[i]->binary_info,
                                                  &info_size) != QNN_SUCCESS) {
            fprintf(stderr, "[sp_qnn] systemContextGetBinaryInfo failed for %s\n", paths[i]);
            goto cleanup_fail;
        }

        slots[i].h         = handles[i];
        slots[i].mu        = &mu;
        slots[i].cv        = &cv;
        slots[i].remaining = &remaining;

        /* binaryBufferSize is const in the struct definition — initialize
         * via designated init through a pointer write rather than assigning. */
        QnnContext_ParamsV1_t v1_init = {
            .config           = per_ctx_cfgs,
            .binaryBuffer     = handles[i]->bin_data,
            .binaryBufferSize = handles[i]->bin_size,
            .profile          = NULL,
            .notifyFunc       = sp_qnn_list_notify,
            .notifyParam      = &slots[i],
        };
        params[i].version = QNN_CONTEXT_PARAMS_VERSION_1;
        memcpy(&params[i].v1, &v1_init, sizeof(v1_init));

        paramPtrs[i] = &params[i];
    }
    paramPtrs[n] = NULL;  /* NULL-terminate */

    /* Fire the async multi-load. */
    fprintf(stderr, "[sp_qnn] createFromBinaryListAsync n=%zu ...\n", n);
    Qnn_ErrorHandle_t rc = g_lib.qnn.contextCreateFromBinaryListAsync(
        g_lib.backend, g_lib.device,
        paramPtrs, list_cfg,
        NULL /*signal*/);
    if (rc != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] createFromBinaryListAsync returned 0x%lx\n",
                (unsigned long)rc);
        goto cleanup_fail;
    }

    /* Join: wait for all notify callbacks to fire. */
    pthread_mutex_lock(&mu);
    while (remaining > 0) {
        pthread_cond_wait(&cv, &mu);
    }
    pthread_mutex_unlock(&mu);

    /* Validate every slot succeeded; if any failed, tear down all. */
    int ok = 1;
    for (size_t i = 0; i < n; ++i) {
        if (slots[i].status != QNN_SUCCESS) {
            fprintf(stderr, "[sp_qnn] slot %zu failed: 0x%lx\n",
                    i, (unsigned long)slots[i].status);
            ok = 0;
        }
    }
    if (!ok) goto cleanup_fail;

    /* All contexts created. Wire each handle, retrieve graph if needed,
     * cache I/O metadata. The async notify already gave us context+graph
     * for each slot, but for safety also call graphRetrieve when the
     * caller specified an explicit name. */
    for (size_t i = 0; i < n; ++i) {
        handles[i]->context = slots[i].ctx;
        handles[i]->graph   = slots[i].graph;

        /* Resolve I/O templates from binary_info. */
        const char *bin_graph_name = NULL;
        const Qnn_Tensor_t *bin_inputs = NULL, *bin_outputs = NULL;
        uint32_t n_in = 0, n_out = 0;
        if (resolve_graph_info(handles[i]->binary_info, &bin_graph_name,
                               &bin_inputs, &n_in,
                               &bin_outputs, &n_out) != 0) {
            fprintf(stderr, "[sp_qnn] resolve_graph_info failed for slot %zu\n", i);
            goto cleanup_fail;
        }
        fprintf(stderr, "[sp_qnn] slot %zu '%s': %u in / %u out\n",
                i, bin_graph_name ? bin_graph_name : "(unnamed)", n_in, n_out);

        /* If the caller supplied a graph name AND the async notify gave us
         * a different / NULL graph, fall back to graphRetrieve. */
        const char *want = (graph_names && graph_names[i]) ? graph_names[i] : bin_graph_name;
        if (!handles[i]->graph && want) {
            if (g_lib.qnn.graphRetrieve(handles[i]->context, want, &handles[i]->graph)
                != QNN_SUCCESS) {
                fprintf(stderr, "[sp_qnn] slot %zu graphRetrieve('%s') failed\n", i, want);
                goto cleanup_fail;
            }
        }

        handles[i]->n_inputs    = n_in;
        handles[i]->n_outputs   = n_out;
        handles[i]->inputs      = calloc(n_in,  sizeof(*handles[i]->inputs));
        handles[i]->outputs     = calloc(n_out, sizeof(*handles[i]->outputs));
        handles[i]->in_tensors  = calloc(n_in,  sizeof(*handles[i]->in_tensors));
        handles[i]->out_tensors = calloc(n_out, sizeof(*handles[i]->out_tensors));
        for (uint32_t k = 0; k < n_in;  ++k) {
            handles[i]->in_tensors[k] = bin_inputs[k];
            cache_tensor_info(&bin_inputs[k], &handles[i]->inputs[k]);
        }
        for (uint32_t k = 0; k < n_out; ++k) {
            handles[i]->out_tensors[k] = bin_outputs[k];
            cache_tensor_info(&bin_outputs[k], &handles[i]->outputs[k]);
        }
        out_handles[i] = handles[i];
    }

    fprintf(stderr, "[sp_qnn] list-load OK: %zu contexts ready (shared HTP resources)\n", n);

    free(slots); free(params); free(paramPtrs); free(handles);
    pthread_mutex_destroy(&mu);
    pthread_cond_destroy(&cv);
    return SP_QNN_OK;

cleanup_fail:
    for (size_t i = 0; i < n; ++i) {
        if (handles[i]) {
            if (handles[i]->context) g_lib.qnn.contextFree(handles[i]->context, NULL);
            if (handles[i]->sysCtx)  g_lib.qsys.systemContextFree(handles[i]->sysCtx);
            unmap_file(handles[i]->bin_data, handles[i]->bin_size);
            free(handles[i]);
            out_handles[i] = NULL;
        }
    }
    free(slots); free(params); free(paramPtrs); free(handles);
    pthread_mutex_destroy(&mu);
    pthread_cond_destroy(&cv);
    return SP_QNN_ERR_CONTEXT_CREATE;
}

/* ------------------------------------------------------------------ */
/* Runtime graph build — bypass the .bin AOT-compile flow entirely.   */
/* Build a single-op MatMul graph at runtime; both inputs are         */
/* APP_WRITE so weights stream in per-execute(), not baked into the   */
/* prepared graph state.                                              */
/* ------------------------------------------------------------------ */

#include "QnnOpDef.h"

/* Set up a Qnn_Tensor_t v2 in our handle's tensor template arrays.
 * Used both for graph registration and per-execute clientBuf binding. */
static void sp_qnn_tensor_init_v2(Qnn_Tensor_t *t, uint32_t id,
                                  const char *name,
                                  Qnn_TensorType_t type,
                                  Qnn_DataType_t dtype,
                                  uint32_t rank, uint32_t *dims) {
    memset(t, 0, sizeof(*t));
    t->version = QNN_TENSOR_VERSION_1;  /* keep aligned with our cache_tensor_info */
    t->v1.id = id;
    t->v1.name = name;
    t->v1.type = type;
    t->v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t->v1.dataType = dtype;
    t->v1.quantizeParams = (Qnn_QuantizeParams_t){
        .encodingDefinition = QNN_DEFINITION_UNDEFINED,
        .quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED,
        .scaleOffsetEncoding = { .scale = 0.0f, .offset = 0 },
    };
    t->v1.rank = rank;
    t->v1.dimensions = dims;
    t->v1.memType = QNN_TENSORMEMTYPE_RAW;
    t->v1.clientBuf.data = NULL;
    t->v1.clientBuf.dataSize = 0;
}

sp_qnn_status sp_qnn_runtime_matmul_create(uint32_t M, uint32_t K, uint32_t N,
                                           uint32_t qnn_dtype,
                                           sp_qnn_handle **out_h) {
    if (!g_lib.initialized || !out_h) return SP_QNN_ERR_INVALID;
    if (sp_qnn_init_device() != SP_QNN_OK) return SP_QNN_ERR_BACKEND_CREATE;

    sp_qnn_handle *h = calloc(1, sizeof(*h));
    if (!h) return SP_QNN_ERR_INVALID;

    /* (1) Empty context attached to shared backend+device. NO binary. */
    if (g_lib.qnn.contextCreate(g_lib.backend, g_lib.device,
                                NULL /*config*/, &h->context) != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] contextCreate (runtime) failed\n");
        free(h);
        return SP_QNN_ERR_CONTEXT_CREATE;
    }

    /* (2) Empty graph in the context. Name doesn't matter for our use. */
    if (g_lib.qnn.graphCreate(h->context, "sp_runtime_matmul",
                              NULL /*config*/, &h->graph) != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] graphCreate (runtime) failed\n");
        g_lib.qnn.contextFree(h->context, NULL);
        free(h);
        return SP_QNN_ERR_GRAPH_RETRIEVE;
    }

    /* (3) Allocate persistent storage for tensor metadata + dim arrays.
     *     The strings + dim arrays MUST outlive the graph — store them
     *     in the handle. We use the existing in_tensors/out_tensors arrays. */
    h->n_inputs   = 2;       /* A and B both APP_WRITE */
    h->n_outputs  = 1;       /* C as APP_READ */
    h->in_tensors  = calloc(2, sizeof(Qnn_Tensor_t));
    h->out_tensors = calloc(1, sizeof(Qnn_Tensor_t));
    h->inputs      = calloc(2, sizeof(sp_qnn_tensor_info));
    h->outputs     = calloc(1, sizeof(sp_qnn_tensor_info));

    /* Dim arrays — leak intentionally on this prototype path. Persistent
     * for the lifetime of the handle (no destroy support yet for runtime
     * graphs; that's fine for the bench harness). bin_data stays NULL so
     * unmap_file in destroy is a no-op. */
    static uint32_t dim_storage[6];
    uint32_t *dims_a = &dim_storage[0]; dims_a[0] = M; dims_a[1] = K;
    uint32_t *dims_b = &dim_storage[2]; dims_b[0] = K; dims_b[1] = N;
    uint32_t *dims_c = &dim_storage[4]; dims_c[0] = M; dims_c[1] = N;

    sp_qnn_tensor_init_v2(&h->in_tensors[0], 0, "in_A",
                          QNN_TENSOR_TYPE_APP_WRITE, qnn_dtype, 2, dims_a);
    sp_qnn_tensor_init_v2(&h->in_tensors[1], 0, "in_B",
                          QNN_TENSOR_TYPE_APP_WRITE, qnn_dtype, 2, dims_b);
    sp_qnn_tensor_init_v2(&h->out_tensors[0], 0, "out_C",
                          QNN_TENSOR_TYPE_APP_READ, qnn_dtype, 2, dims_c);

    /* (4) Register tensors with the graph. The backend assigns each tensor
     *     a unique id which we'll need for opConfig wiring. */
    if (g_lib.qnn.tensorCreateGraphTensor(h->graph, &h->in_tensors[0]) != QNN_SUCCESS
        || g_lib.qnn.tensorCreateGraphTensor(h->graph, &h->in_tensors[1]) != QNN_SUCCESS
        || g_lib.qnn.tensorCreateGraphTensor(h->graph, &h->out_tensors[0]) != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] tensorCreateGraphTensor failed\n");
        sp_qnn_destroy(&h);
        return SP_QNN_ERR_INVALID;
    }

    /* (5) Wire MatMul op as a single graph node. Inputs reference
     *     in_tensors[0..1] by their backend-assigned ids. */
    Qnn_Tensor_t op_inputs[2]  = { h->in_tensors[0], h->in_tensors[1] };
    Qnn_Tensor_t op_outputs[1] = { h->out_tensors[0] };

    Qnn_OpConfig_t opConfig = {
        .version = QNN_OPCONFIG_VERSION_1,
        .v1 = {
            .name             = "matmul_node",
            .packageName      = QNN_OP_PACKAGE_NAME_QTI_AISW,
            .typeName         = QNN_OP_MAT_MUL,
            .numOfParams      = 0,
            .params           = NULL,
            .numOfInputs      = 2,
            .inputTensors     = op_inputs,
            .numOfOutputs     = 1,
            .outputTensors    = op_outputs,
        }
    };

    if (g_lib.qnn.graphAddNode(h->graph, opConfig) != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] graphAddNode(MatMul) failed\n");
        sp_qnn_destroy(&h);
        return SP_QNN_ERR_INVALID;
    }

    /* (6) Finalize the graph — backend compiles to HTP kernels. */
    fprintf(stderr, "[sp_qnn] runtime graphFinalize ...\n");
    Qnn_ErrorHandle_t rc = g_lib.qnn.graphFinalize(h->graph,
                                                    NULL /*profile*/,
                                                    NULL /*signal*/);
    if (rc != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] graphFinalize failed: 0x%lx\n",
                (unsigned long)rc);
        sp_qnn_destroy(&h);
        return SP_QNN_ERR_INVALID;
    }

    /* (7) Cache I/O metadata for sp_qnn_get_io_info / sp_qnn_execute. */
    cache_tensor_info(&h->in_tensors[0],  &h->inputs[0]);
    cache_tensor_info(&h->in_tensors[1],  &h->inputs[1]);
    cache_tensor_info(&h->out_tensors[0], &h->outputs[0]);

    fprintf(stderr, "[sp_qnn] runtime MatMul graph ready: A[%u,%u] x B[%u,%u] = C[%u,%u] dtype=%u\n",
            M, K, K, N, M, N, qnn_dtype);
    *out_h = h;
    return SP_QNN_OK;
}

/* Multi-op runtime graph: Q[M,K] @ K^T[K,N] -> kq_logits[M,N] -> Softmax -> attn_weights[M,N].
 *
 * Three persistent dim arrays in static storage (handle prototype path —
 * lifetime is process-scoped which is fine for the bench harness). */
sp_qnn_status sp_qnn_runtime_kq_softmax_create(uint32_t M_q,
                                               uint32_t K_dim,
                                               uint32_t N_kv,
                                               uint32_t qnn_dtype,
                                               sp_qnn_handle **out_h) {
    if (!g_lib.initialized || !out_h) return SP_QNN_ERR_INVALID;
    if (sp_qnn_init_device() != SP_QNN_OK) return SP_QNN_ERR_BACKEND_CREATE;

    sp_qnn_handle *h = calloc(1, sizeof(*h));
    if (!h) return SP_QNN_ERR_INVALID;

    if (g_lib.qnn.contextCreate(g_lib.backend, g_lib.device, NULL, &h->context) != QNN_SUCCESS) {
        free(h); return SP_QNN_ERR_CONTEXT_CREATE;
    }
    if (g_lib.qnn.graphCreate(h->context, "sp_runtime_kq_softmax",
                              NULL, &h->graph) != QNN_SUCCESS) {
        g_lib.qnn.contextFree(h->context, NULL); free(h);
        return SP_QNN_ERR_GRAPH_RETRIEVE;
    }

    /* Tensors:
     *   in[0] Q          APP_WRITE   [M_q, K_dim]
     *   in[1] K          APP_WRITE   [N_kv, K_dim]   (we MatMul-transpose-in1)
     *   mid[0] kq_logits NATIVE      [M_q, N_kv]
     *   out[0] attn      APP_READ    [M_q, N_kv]
     */
    h->n_inputs   = 2;
    h->n_outputs  = 1;
    h->in_tensors  = calloc(2, sizeof(Qnn_Tensor_t));
    h->out_tensors = calloc(1, sizeof(Qnn_Tensor_t));
    h->inputs      = calloc(2, sizeof(sp_qnn_tensor_info));
    h->outputs     = calloc(1, sizeof(sp_qnn_tensor_info));
    Qnn_Tensor_t mid_kq_logits;
    memset(&mid_kq_logits, 0, sizeof(mid_kq_logits));

    static uint32_t kq_dims[8];
    uint32_t *dims_q   = &kq_dims[0]; dims_q[0]   = M_q;  dims_q[1]   = K_dim;
    uint32_t *dims_k   = &kq_dims[2]; dims_k[0]   = N_kv; dims_k[1]   = K_dim;
    uint32_t *dims_log = &kq_dims[4]; dims_log[0] = M_q;  dims_log[1] = N_kv;
    uint32_t *dims_atn = &kq_dims[6]; dims_atn[0] = M_q;  dims_atn[1] = N_kv;

    sp_qnn_tensor_init_v2(&h->in_tensors[0], 0, "kq_Q",
                          QNN_TENSOR_TYPE_APP_WRITE, qnn_dtype, 2, dims_q);
    sp_qnn_tensor_init_v2(&h->in_tensors[1], 0, "kq_K",
                          QNN_TENSOR_TYPE_APP_WRITE, qnn_dtype, 2, dims_k);
    sp_qnn_tensor_init_v2(&mid_kq_logits, 0, "kq_logits",
                          QNN_TENSOR_TYPE_NATIVE, qnn_dtype, 2, dims_log);
    sp_qnn_tensor_init_v2(&h->out_tensors[0], 0, "kq_attn",
                          QNN_TENSOR_TYPE_APP_READ, qnn_dtype, 2, dims_atn);

    if (g_lib.qnn.tensorCreateGraphTensor(h->graph, &h->in_tensors[0]) != QNN_SUCCESS
     || g_lib.qnn.tensorCreateGraphTensor(h->graph, &h->in_tensors[1]) != QNN_SUCCESS
     || g_lib.qnn.tensorCreateGraphTensor(h->graph, &mid_kq_logits)    != QNN_SUCCESS
     || g_lib.qnn.tensorCreateGraphTensor(h->graph, &h->out_tensors[0]) != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] kq tensorCreateGraphTensor failed\n");
        sp_qnn_destroy(&h); return SP_QNN_ERR_INVALID;
    }

    /* Op 1: MatMul with transpose_in1=true → Q @ K^T. */
    static uint32_t param_dim_scalar[1] = {1};
    Qnn_Param_t mm_params[2];
    memset(mm_params, 0, sizeof(mm_params));
    mm_params[0].paramType    = QNN_PARAMTYPE_SCALAR;
    mm_params[0].name         = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0;
    mm_params[0].scalarParam.dataType  = QNN_DATATYPE_BOOL_8;
    mm_params[0].scalarParam.bool8Value = 0;
    mm_params[1].paramType    = QNN_PARAMTYPE_SCALAR;
    mm_params[1].name         = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1;
    mm_params[1].scalarParam.dataType  = QNN_DATATYPE_BOOL_8;
    mm_params[1].scalarParam.bool8Value = 1;
    (void)param_dim_scalar;

    Qnn_Tensor_t mm_inputs[2]  = { h->in_tensors[0], h->in_tensors[1] };
    Qnn_Tensor_t mm_outputs[1] = { mid_kq_logits };

    Qnn_OpConfig_t mm_cfg = {
        .version = QNN_OPCONFIG_VERSION_1,
        .v1 = {
            .name             = "kq_matmul",
            .packageName      = QNN_OP_PACKAGE_NAME_QTI_AISW,
            .typeName         = QNN_OP_MAT_MUL,
            .numOfParams      = 2,
            .params           = mm_params,
            .numOfInputs      = 2,
            .inputTensors     = mm_inputs,
            .numOfOutputs     = 1,
            .outputTensors    = mm_outputs,
        }
    };
    if (g_lib.qnn.graphAddNode(h->graph, mm_cfg) != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] kq_matmul addNode failed\n");
        sp_qnn_destroy(&h); return SP_QNN_ERR_INVALID;
    }

    /* Op 2: Softmax over the last dim (n_kv). QNN's Softmax has an "axis"
     * scalar param. For 2D [M_q, N_kv] axis=1 = last dim. */
    Qnn_Param_t sm_params[1];
    memset(sm_params, 0, sizeof(sm_params));
    sm_params[0].paramType   = QNN_PARAMTYPE_SCALAR;
    sm_params[0].name        = "axis";
    sm_params[0].scalarParam.dataType    = QNN_DATATYPE_UINT_32;
    sm_params[0].scalarParam.uint32Value = 1;

    Qnn_Tensor_t sm_inputs[1]  = { mid_kq_logits };
    Qnn_Tensor_t sm_outputs[1] = { h->out_tensors[0] };

    Qnn_OpConfig_t sm_cfg = {
        .version = QNN_OPCONFIG_VERSION_1,
        .v1 = {
            .name             = "kq_softmax",
            .packageName      = QNN_OP_PACKAGE_NAME_QTI_AISW,
            .typeName         = "Softmax",
            .numOfParams      = 1,
            .params           = sm_params,
            .numOfInputs      = 1,
            .inputTensors     = sm_inputs,
            .numOfOutputs     = 1,
            .outputTensors    = sm_outputs,
        }
    };
    if (g_lib.qnn.graphAddNode(h->graph, sm_cfg) != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] kq_softmax addNode failed\n");
        sp_qnn_destroy(&h); return SP_QNN_ERR_INVALID;
    }

    fprintf(stderr, "[sp_qnn] runtime KQ+Softmax graph: finalize ...\n");
    if (g_lib.qnn.graphFinalize(h->graph, NULL, NULL) != QNN_SUCCESS) {
        fprintf(stderr, "[sp_qnn] graphFinalize (KQ+Softmax) failed\n");
        sp_qnn_destroy(&h); return SP_QNN_ERR_INVALID;
    }

    cache_tensor_info(&h->in_tensors[0],  &h->inputs[0]);
    cache_tensor_info(&h->in_tensors[1],  &h->inputs[1]);
    cache_tensor_info(&h->out_tensors[0], &h->outputs[0]);

    fprintf(stderr, "[sp_qnn] runtime KQ+Softmax graph ready: Q[%u,%u]@K^T[%u,%u]->softmax->attn[%u,%u]\n",
            M_q, K_dim, N_kv, K_dim, M_q, N_kv);
    *out_h = h;
    return SP_QNN_OK;
}

/* Register a persistent input tensor via QnnMem_register. The data buffer
 * is bound once (memHandle stored in the tensor template) and not rebound
 * on each execute. Caller owns the buffer; must outlive sp_qnn_destroy(). */
sp_qnn_status sp_qnn_register_persistent_input(sp_qnn_handle *h,
                                               size_t tensor_idx,
                                               void *data,
                                               size_t bytes) {
    if (!h || tensor_idx >= h->n_inputs || !data || !bytes)
        return SP_QNN_ERR_INVALID;
    /* QnnMem_register is the canonical path. We use a "rawMem" descriptor
     * pointing at our buffer. Some backends require rpcmem for this; our
     * V69 HTP accepts plain anon mem, with an internal DMA copy on first
     * use. The savings here are skipping the per-execute clientBuf rebind. */
    /* On V69 with QAIRT 2.45 the memRegister(CUSTOM, NULL_customInfo) path
     * crashes inside libQnnHtp (segfaults on NULL deref). Production needs
     * rpcmem-allocated ION fd to populate ionInfo properly. For our test
     * harness, skip the memRegister and use the SOFTWARE pseudo-persistence
     * path directly: stash the data pointer in clientBuf, then sp_qnn_execute
     * sees NULL caller-data for this tensor and preserves the prior binding.
     *
     * Net effect: we still re-validate the clientBuf pointer per call (cheap),
     * but skip ANY copy of the K bytes themselves. The proper memhandle
     * optimization needs rpcmem plumbed in (Phase 2.5b). */
    h->in_tensors[tensor_idx].v1.memType = QNN_TENSORMEMTYPE_RAW;
    h->in_tensors[tensor_idx].v1.clientBuf.data     = data;
    h->in_tensors[tensor_idx].v1.clientBuf.dataSize = (uint32_t)bytes;
    fprintf(stderr, "[sp_qnn] tensor %zu pseudo-persistent: clientBuf preserved across exec calls "
                    "(memhandle path needs rpcmem fd, deferred to Phase 2.5b)\n",
            tensor_idx);
    return SP_QNN_OK;
}

/* Phase 2.6b — allocate an rpcmem-ION buffer and bind it as a persistent
 * QNN MemHandle to the given input tensor. See sp_qnn.h for the full
 * mechanism. The allocation is tracked in h->persistent[] so that
 * sp_qnn_destroy() will deregister + rpcmem_free in the right order. */
sp_qnn_status sp_qnn_alloc_persistent(sp_qnn_handle *h,
                                      size_t tensor_idx,
                                      size_t bytes,
                                      void **out_user_ptr) {
    if (!h || tensor_idx >= h->n_inputs || !bytes || !out_user_ptr)
        return SP_QNN_ERR_INVALID;
    if (h->n_persistent >= SP_QNN_MAX_PERSISTENT) {
        fprintf(stderr, "[sp_qnn] persistent table full (max %d)\n",
                SP_QNN_MAX_PERSISTENT);
        return SP_QNN_ERR_INVALID;
    }
    if (!g_lib.qnn.memRegister || !g_lib.qnn.memDeRegister) {
        fprintf(stderr, "[sp_qnn] QNN interface lacks memRegister — cannot do ION persistence\n");
        return SP_QNN_ERR_INVALID;
    }
    if (sp_qnn_rpcmem_load() != 0) {
        return SP_QNN_ERR_DLOPEN;
    }

    /* Extract dims + dtype from the tensor template (v1 vs v2). */
    uint32_t rank = 0;
    const uint32_t *dims = NULL;
    Qnn_DataType_t dtype = QNN_DATATYPE_UNDEFINED;
    Qnn_Tensor_t *t = &h->in_tensors[tensor_idx];
    if (t->version == QNN_TENSOR_VERSION_1) {
        rank  = t->v1.rank;
        dims  = t->v1.dimensions;
        dtype = t->v1.dataType;
    } else {
        rank  = t->v2.rank;
        dims  = t->v2.dimensions;
        dtype = t->v2.dataType;
    }
    if (!rank || !dims) {
        fprintf(stderr, "[sp_qnn] tensor %zu has no rank/dims — alloc_persistent unsafe\n",
                tensor_idx);
        return SP_QNN_ERR_TENSOR_SHAPE;
    }

    /* Step 1: rpcmem_alloc — page-aligned ION-backed buffer. */
    void *user_ptr = g_rpc.alloc(SP_RPCMEM_HEAP_ID_SYSTEM,
                                 SP_RPCMEM_DEFAULT_FLAGS, bytes);
    if (!user_ptr) {
        fprintf(stderr, "[sp_qnn] rpcmem_alloc(%zu bytes) failed\n", bytes);
        return SP_QNN_ERR_INVALID;
    }
    int ion_fd = g_rpc.to_fd(user_ptr);
    if (ion_fd < 0) {
        fprintf(stderr, "[sp_qnn] rpcmem_to_fd failed (rc=%d)\n", ion_fd);
        g_rpc.free_(user_ptr);
        return SP_QNN_ERR_INVALID;
    }

    /* Step 2: build a Qnn_MemDescriptor_t describing those bytes as a
     * tensor of the same shape/dtype as our tensor template. */
    Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
    desc.memShape.numDim       = rank;
    desc.memShape.dimSize      = (uint32_t *)dims;     /* dims live in template */
    desc.memShape.shapeConfig  = NULL;
    desc.dataType              = dtype;
    desc.memType               = QNN_MEM_TYPE_ION;
    desc.ionInfo.fd            = ion_fd;

    /* Step 3: register with the context — returns an opaque MemHandle. */
    Qnn_MemHandle_t mh = NULL;
    Qnn_ErrorHandle_t rc = g_lib.qnn.memRegister(h->context, &desc, 1, &mh);
    if (rc != QNN_SUCCESS || !mh) {
        fprintf(stderr, "[sp_qnn] QnnMem_register failed: 0x%lx (tensor %zu, %zu B)\n",
                (unsigned long)rc, tensor_idx, bytes);
        g_rpc.free_(user_ptr);
        return SP_QNN_ERR_INVALID;
    }

    /* Step 4: stamp the handle into the tensor template. The union member
     * memHandle overlaps clientBuf, so once memType=MEMHANDLE the prior
     * clientBuf is naturally ignored. tensor_set_buf already early-exits
     * when memType==MEMHANDLE so subsequent execute() calls don't clobber. */
    if (t->version == QNN_TENSOR_VERSION_1) {
        t->v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
        t->v1.memHandle = mh;
    } else {
        t->v2.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
        t->v2.memHandle = mh;
    }

    /* Track for destroy(). */
    sp_qnn_persistent_t *p = &h->persistent[h->n_persistent++];
    p->tensor_idx = tensor_idx;
    p->user_ptr   = user_ptr;
    p->mem_handle = mh;
    p->bytes      = bytes;

    fprintf(stderr, "[sp_qnn] tensor %zu persistent ION-bound: %zu B @ %p (fd=%d, handle=%p)\n",
            tensor_idx, bytes, user_ptr, ion_fd, (void *)mh);

    *out_user_ptr = user_ptr;
    return SP_QNN_OK;
}
