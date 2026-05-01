/* Implementation of sp_llama_qnn_kq_dispatch. Backed by sp_qnn runtime
 * graph build. See sp_llama_qnn.h for design notes. */
#include "sp_llama_qnn.h"
#include "sp_qnn.h"
#include "QnnTypes.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SP_LLAMA_QNN_KQ_CACHE_MAX 64  /* unique shapes per session */

typedef struct {
    uint32_t M_q, K_dim, N_kv;
    sp_qnn_handle *h;
    int            k_bound;
    void          *bound_k_ptr;  /* track which K is currently bound */

    /* Phase 2.6b: ION-backed persistent K buffer. When non-NULL, K is
     * memHandle-bound to this rpcmem-allocated ION buffer; per-dispatch
     * we memcpy caller's K into it (cheap — same physical pages mapped
     * into HTP DMA, no marshal). NULL = legacy clientBuf-preservation
     * fallback (e.g., rpcmem load failed). */
    void  *ion_k_ptr;
    size_t ion_k_bytes;
} kq_slot_t;

struct sp_llama_qnn_kq_cache {
    kq_slot_t       slots[SP_LLAMA_QNN_KQ_CACHE_MAX];
    int             n_slots;
    pthread_mutex_t mu;
    int             qnn_init_done;
};

sp_llama_qnn_kq_cache *sp_llama_qnn_kq_cache_create(void) {
    sp_llama_qnn_kq_cache *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    pthread_mutex_init(&c->mu, NULL);
    return c;
}

void sp_llama_qnn_kq_cache_destroy(sp_llama_qnn_kq_cache **c_io) {
    if (!c_io || !*c_io) return;
    sp_llama_qnn_kq_cache *c = *c_io;
    pthread_mutex_lock(&c->mu);
    for (int i = 0; i < c->n_slots; ++i) {
        if (c->slots[i].h) sp_qnn_destroy(&c->slots[i].h);
    }
    pthread_mutex_unlock(&c->mu);
    pthread_mutex_destroy(&c->mu);
    if (c->qnn_init_done) sp_qnn_shutdown();
    free(c);
    *c_io = NULL;
}

/* Find or lazily create the slot for this shape. Caller holds c->mu. */
static kq_slot_t *find_or_create_slot(sp_llama_qnn_kq_cache *c,
                                       uint32_t M_q, uint32_t K_dim, uint32_t N_kv) {
    for (int i = 0; i < c->n_slots; ++i) {
        if (c->slots[i].M_q == M_q && c->slots[i].K_dim == K_dim
            && c->slots[i].N_kv == N_kv) {
            return &c->slots[i];
        }
    }
    if (c->n_slots >= SP_LLAMA_QNN_KQ_CACHE_MAX) {
        fprintf(stderr, "[sp_llama_qnn] cache full at %d shapes\n", c->n_slots);
        return NULL;
    }
    /* First-ever call: bring up sp_qnn library state. */
    if (!c->qnn_init_done) {
        if (sp_qnn_init(NULL, NULL) != SP_QNN_OK) {
            fprintf(stderr, "[sp_llama_qnn] sp_qnn_init failed\n");
            return NULL;
        }
        c->qnn_init_done = 1;
    }

    kq_slot_t *s = &c->slots[c->n_slots];
    s->M_q   = M_q;
    s->K_dim = K_dim;
    s->N_kv  = N_kv;
    s->h     = NULL;
    s->k_bound = 0;
    s->bound_k_ptr = NULL;
    s->ion_k_ptr   = NULL;
    s->ion_k_bytes = 0;

    fprintf(stderr, "[sp_llama_qnn] new shape: Q[%u,%u]@K^T[%u,%u]->softmax->[%u,%u] (graphFinalize ~50ms)\n",
            M_q, K_dim, N_kv, K_dim, M_q, N_kv);
    if (sp_qnn_runtime_kq_softmax_create(M_q, K_dim, N_kv,
                                          QNN_DATATYPE_FLOAT_16,
                                          &s->h) != SP_QNN_OK) {
        fprintf(stderr, "[sp_llama_qnn] kq_softmax_create failed for shape\n");
        return NULL;
    }

    /* Phase 2.6b: try to allocate a persistent ION-backed K buffer for
     * this shape. K bytes = N_kv * K_dim * 2 (fp16). If this succeeds,
     * subsequent dispatches memcpy K into the ION pages and skip rebind.
     * If it fails (no rpcmem available, e.g. desktop test), we fall back
     * to the legacy clientBuf-preservation path on first dispatch. */
    const size_t k_bytes = (size_t)N_kv * (size_t)K_dim * 2;
    void *ion_ptr = NULL;
    if (sp_qnn_alloc_persistent(s->h, /*tensor_idx=*/1, k_bytes, &ion_ptr) == SP_QNN_OK
        && ion_ptr != NULL) {
        s->ion_k_ptr   = ion_ptr;
        s->ion_k_bytes = k_bytes;
        fprintf(stderr, "[sp_llama_qnn] persistent K (%zu B) ION-bound for shape\n",
                k_bytes);
    } else {
        fprintf(stderr, "[sp_llama_qnn] persistent K alloc failed — falling back to per-call rebind\n");
    }

    c->n_slots++;
    return s;
}

int sp_llama_qnn_kq_dispatch(sp_llama_qnn_kq_cache *cache,
                             uint32_t M_q,
                             uint32_t K_dim,
                             uint32_t N_kv,
                             const void *q_data, size_t q_bytes,
                             void *k_data,       size_t k_bytes,
                             void *attn_out,     size_t out_bytes,
                             uint64_t *exec_us) {
    if (!cache || !q_data || !k_data || !attn_out) return -1;

    /* One-shot debug log so we can confirm dispatch actually fires inside
     * llama-cli (vs the wrapper's gate-condition short-circuiting in
     * llama_sp_fused_kq.cpp). Phase 2.6b debug aid. */
    static int s_first_dispatch_logged = 0;
    if (!s_first_dispatch_logged) {
        s_first_dispatch_logged = 1;
        fprintf(stderr, "[sp_llama_qnn] FIRST DISPATCH: M_q=%u K_dim=%u N_kv=%u\n",
                M_q, K_dim, N_kv);
    }

    pthread_mutex_lock(&cache->mu);
    kq_slot_t *s = find_or_create_slot(cache, M_q, K_dim, N_kv);
    if (!s) { pthread_mutex_unlock(&cache->mu); return -2; }

    /* Phase 2.6b: K binding has two paths.
     *
     * (A) ION-backed persistent (preferred): copy caller's K bytes into
     *     the rpcmem ION buffer that QNN MemHandle is already bound to.
     *     The HTP sees the same physical pages — no rebind, no marshal.
     *     The memcpy itself stays on the host CPU but lands in pages
     *     that are also mapped into the SMMU view, so it's effectively
     *     "DMA-coherent" by virtue of being the same memory.
     *
     * (B) Legacy fallback: clientBuf preservation. We only rebind when
     *     k_data pointer changes (new token / new KV row). When the same
     *     pointer is reused, the prior binding is kept (NULL passed to
     *     execute()). This is a perf trick — there's no real persistence
     *     across token boundaries.
     */
    if (s->ion_k_ptr) {
        /* (A) */
        if (k_bytes > s->ion_k_bytes) {
            fprintf(stderr, "[sp_llama_qnn] k_bytes=%zu > ion buffer %zu — bug\n",
                    k_bytes, s->ion_k_bytes);
            pthread_mutex_unlock(&cache->mu);
            return -4;
        }
        memcpy(s->ion_k_ptr, k_data, k_bytes);
    } else {
        /* (B) */
        if (!s->k_bound || s->bound_k_ptr != k_data) {
            sp_qnn_register_persistent_input(s->h, /*tensor_idx=*/1, k_data, k_bytes);
            s->k_bound = 1;
            s->bound_k_ptr = k_data;
        }
    }

    /* Per-call inputs: Q always rebinds (RAW clientBuf), K is NULL because
     * either MemHandle (path A) or preserved clientBuf (path B) — both
     * cause tensor_set_buf to skip and reuse the prior binding. */
    const void *ins[]  = { q_data, NULL };
    const size_t in_sz[] = { q_bytes, k_bytes };
    void *outs[] = { attn_out };
    const size_t out_sz[] = { out_bytes };

    sp_qnn_status rc = sp_qnn_execute(s->h, ins, in_sz, outs, out_sz, exec_us);
    pthread_mutex_unlock(&cache->mu);
    return (rc == SP_QNN_OK) ? 0 : -3;
}
