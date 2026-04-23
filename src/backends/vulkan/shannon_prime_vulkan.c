// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

// Vulkan compute backend for VHT2 KV cache compression.
//
// Two build modes:
//   1. SHANNON_PRIME_VULKAN_ENABLED=1 (Makefile auto-detects VULKAN_SDK):
//      Real GPU dispatch through four compute pipelines —
//        vilenkin         : in-place staged Hartley, self-inverse VHT2
//        mobius_reorder   : squarefree-first permutation
//        band_quantize    : per-band abs-max int8 quantisation
//        band_dequantize  : inverse of quantize (new shader)
//      Compressed cache bytes live in host-visible GPU memory; the host
//      memcpy's in/out of the mapped scratch buffers for each op.
//   2. Default (no SDK): CPU fallback through sp_shadow_* which routes
//      to the core staged VHT2 (sp_vht2_forward_f32). sqfree dims work.

#define _POSIX_C_SOURCE 199309L

#include "shannon_prime_vulkan.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef SHANNON_PRIME_VULKAN_ENABLED
#include <vulkan/vulkan.h>
#endif

// ============================================================================
// Internal structures
// ============================================================================

#ifdef SHANNON_PRIME_VULKAN_ENABLED

typedef struct {
    VkInstance       instance;
    VkPhysicalDevice phys_device;
    VkDevice         device;
    VkQueue          queue;
    uint32_t         queue_family;
    int              owns_device;

    VkCommandPool    cmd_pool;
    VkCommandBuffer  cmd_buf;     // reusable
    VkFence          fence;

    // Shader modules + pipelines
    VkShaderModule sm_vilenkin, sm_mobius, sm_bquant, sm_bdequant;
    VkDescriptorSetLayout dsl_1, dsl_3, dsl_2;
    VkPipelineLayout pl_1, pl_3, pl_2;
    VkPipeline pipe_vilenkin;      // 1 SSBO (in-place)
    VkPipeline pipe_mobius;        // 3 SSBO (in, out, order)
    VkPipeline pipe_bquant;        // 2 SSBO (in float, out uint)
    VkPipeline pipe_bdequant;      // 2 SSBO (in uint, out float)

    VkDescriptorPool desc_pool;

    // Buffers (host-visible coherent for simplicity)
    // scratch_a, scratch_b : pad_dim * float32 (ping-pong for reorder)
    // buf_quant             : max(k_bytes, v_bytes) bytes (packed output)
    // buf_mobius_order      : pad_dim ints
    // buf_factors           : up to 8 ints (prime factorisation)
    VkBuffer buf_a, buf_b, buf_quant, buf_mobius_order;
    VkDeviceMemory mem_a, mem_b, mem_quant, mem_mobius_order;
    void *map_a, *map_b, *map_quant, *map_mobius_order;

    // Descriptor sets (reused across calls)
    VkDescriptorSet ds_vil_a, ds_vil_b;                 // vilenkin in-place on a / b
    VkDescriptorSet ds_mob_a_to_b, ds_mob_b_to_a;       // mobius a->b / b->a
    VkDescriptorSet ds_quant_b_to_quant;                // quantize b->buf_quant
    VkDescriptorSet ds_dequant_quant_to_a;              // dequantize buf_quant->a
} sp_vk_impl_t;

#endif // SHANNON_PRIME_VULKAN_ENABLED

struct sp_vulkan_cache_s {
    sp_config_t       config;
    sp_band_config_t  k_bands;
    sp_band_config_t  v_bands;
    sp_mobius_mask_t   mobius_mask;
    int               max_seq_len;

    // Cache storage — CPU-side bytes (compressed). GPU is the compute engine,
    // not the long-term store, so per-slot compressed bytes live here.
    uint8_t **k_cache;
    uint8_t **v_cache;
    // Per-vector blob stride in the K/V cache. For the default CPU path this
    // equals the corresponding `bands->total_bytes` (fp16 scale + uint8
    // packed). When SHANNON_PRIME_VULKAN_FORCE_GPU=1 is set at init time
    // the shaders emit a fp32-scale + uint32-packed layout that needs more
    // room (e.g. 84 bytes vs 76 bytes for 5,5,4,3@hd=128), so we bump the
    // slot stride to max(cpu_bytes, gpu_bytes) and all memcpys use this
    // field instead of the raw `total_bytes`. A single cache can't mix
    // both formats — the choice is frozen at init.
    int k_blob_bytes;
    int v_blob_bytes;
    int force_gpu_blob;  // 1 when slots are allocated in GPU format

    // Either GPU impl or CPU fallback
    int use_cpu_fallback;
    sp_shadow_cache_t cpu_cache;

#ifdef SHANNON_PRIME_VULKAN_ENABLED
    sp_vk_impl_t vk;
    int vk_live;   // set once init_vulkan_pipelines succeeds
#endif
};

// ============================================================================
// CPU Fallback (used when Vulkan unavailable)
// ============================================================================

static int init_cpu_fallback(sp_vulkan_cache_t *cc, const sp_config_t *cfg,
                             int max_seq_len) {
    cc->use_cpu_fallback = 1;
    if (sp_shadow_cache_init(&cc->cpu_cache, cfg) != 0) return -1;

    int n_slots = cfg->n_layers * cfg->n_heads_kv;
    cc->cpu_cache.k_cache = (uint8_t **)calloc(n_slots, sizeof(uint8_t *));
    cc->cpu_cache.v_cache = (uint8_t **)calloc(n_slots, sizeof(uint8_t *));
    for (int i = 0; i < n_slots; i++) {
        cc->cpu_cache.k_cache[i] = (uint8_t *)calloc(
            max_seq_len, cc->cpu_cache.k_bands.total_bytes);
        cc->cpu_cache.v_cache[i] = (uint8_t *)calloc(
            max_seq_len, cc->cpu_cache.v_bands.total_bytes);
    }
    fprintf(stderr, "[Shannon-Prime Vulkan] Using CPU fallback\n");
    return 0;
}

static void free_cpu_fallback(sp_vulkan_cache_t *cc) {
    if (!cc->use_cpu_fallback) return;
    int n_slots = cc->config.n_layers * cc->config.n_heads_kv;
    for (int i = 0; i < n_slots; i++) {
        free(cc->cpu_cache.k_cache[i]);
        free(cc->cpu_cache.v_cache[i]);
    }
    free(cc->cpu_cache.k_cache);
    free(cc->cpu_cache.v_cache);
    sp_shadow_cache_free(&cc->cpu_cache);
}

// ============================================================================
// Vulkan GPU implementation
// ============================================================================

#ifdef SHANNON_PRIME_VULKAN_ENABLED

// Small helper — check Vulkan result, log + return -1 on failure
#define VK_OK(expr) do { VkResult _r = (expr); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "[Vulkan] %s failed: %d\n", #expr, (int)_r); return -1; } } while (0)

static int vk_read_spv(const char *path, uint32_t **out, size_t *out_bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz % 4 != 0) { fclose(f); return -1; }
    uint32_t *buf = (uint32_t *)malloc(sz);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, sz, f) != (size_t)sz) { free(buf); fclose(f); return -1; }
    fclose(f);
    *out = buf;
    *out_bytes = (size_t)sz;
    return 0;
}

static const char *vk_shader_dir(void) {
    const char *env = getenv("SHANNON_PRIME_VULKAN_SHADER_DIR");
    return env ? env : "build/shaders";
}

static int vk_load_shader(VkDevice dev, const char *name, VkShaderModule *out) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.spv", vk_shader_dir(), name);
    uint32_t *code = NULL; size_t bytes = 0;
    if (vk_read_spv(path, &code, &bytes) != 0) {
        fprintf(stderr, "[Vulkan] failed to read %s\n", path);
        return -1;
    }
    VkShaderModuleCreateInfo info = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    info.codeSize = bytes;
    info.pCode = code;
    VkResult r = vkCreateShaderModule(dev, &info, NULL, out);
    free(code);
    if (r != VK_SUCCESS) { fprintf(stderr, "[Vulkan] vkCreateShaderModule(%s) = %d\n", name, (int)r); return -1; }
    return 0;
}

static int vk_find_memtype(VkPhysicalDevice pd, uint32_t type_bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(pd, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags & want) == want) {
            return (int)i;
        }
    }
    return -1;
}

static int vk_alloc_buffer(sp_vk_impl_t *vk, VkDeviceSize size,
                           VkBuffer *out_buf, VkDeviceMemory *out_mem,
                           void **out_map) {
    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
             | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
             | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_OK(vkCreateBuffer(vk->device, &bi, NULL, out_buf));

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(vk->device, *out_buf, &mr);
    int mt = vk_find_memtype(vk->phys_device, mr.memoryTypeBits,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt < 0) return -1;

    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = (uint32_t)mt;
    VK_OK(vkAllocateMemory(vk->device, &mai, NULL, out_mem));
    VK_OK(vkBindBufferMemory(vk->device, *out_buf, *out_mem, 0));
    VK_OK(vkMapMemory(vk->device, *out_mem, 0, size, 0, out_map));
    return 0;
}

static int vk_create_dsl(VkDevice dev, uint32_t n_bindings,
                         VkDescriptorSetLayout *out) {
    VkDescriptorSetLayoutBinding binds[8];
    for (uint32_t i = 0; i < n_bindings; i++) {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        binds[i].pImmutableSamplers = NULL;
    }
    VkDescriptorSetLayoutCreateInfo ci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    ci.bindingCount = n_bindings;
    ci.pBindings = binds;
    VK_OK(vkCreateDescriptorSetLayout(dev, &ci, NULL, out));
    return 0;
}

static int vk_create_pipeline_layout(VkDevice dev, VkDescriptorSetLayout dsl,
                                     uint32_t push_bytes,
                                     VkPipelineLayout *out) {
    VkPushConstantRange pc = { VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes };
    VkPipelineLayoutCreateInfo ci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    ci.setLayoutCount = 1;
    ci.pSetLayouts = &dsl;
    ci.pushConstantRangeCount = push_bytes ? 1 : 0;
    ci.pPushConstantRanges = push_bytes ? &pc : NULL;
    VK_OK(vkCreatePipelineLayout(dev, &ci, NULL, out));
    return 0;
}

static int vk_create_compute_pipeline(VkDevice dev, VkShaderModule sm,
                                      VkPipelineLayout pl, VkPipeline *out) {
    VkPipelineShaderStageCreateInfo stage = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = sm;
    stage.pName = "main";
    VkComputePipelineCreateInfo ci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    ci.stage = stage;
    ci.layout = pl;
    VK_OK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, out));
    return 0;
}

static void vk_update_ds(VkDevice dev, VkDescriptorSet ds, uint32_t binding,
                         VkBuffer buf, VkDeviceSize size) {
    VkDescriptorBufferInfo bi = { buf, 0, size };
    VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    w.dstSet = ds;
    w.dstBinding = binding;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &bi;
    vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
}

// Factor n into {2,3,5,7,11}. Up to 8 factors (hd=256 = 2^8 is max).
// Returns count or -1 if n has prime factor > 11.
static int vk_factor_small(int n, uint32_t *factors_out) {
    static const int primes[] = {2, 3, 5, 7, 11};
    int nf = 0;
    int d = n;
    for (int i = 0; i < 5; i++) {
        while (d % primes[i] == 0 && nf < 8) {
            factors_out[nf++] = (uint32_t)primes[i];
            d /= primes[i];
        }
    }
    return (d == 1) ? nf : -1;
}

static int init_vulkan_pipelines(sp_vulkan_cache_t *cc, void *user_device,
                                 void *user_queue, int gpu_index) {
    sp_vk_impl_t *vk = &cc->vk;
    memset(vk, 0, sizeof(*vk));

    if (user_device && user_queue) {
        vk->device = (VkDevice)user_device;
        vk->queue  = (VkQueue)user_queue;
        vk->owns_device = 0;
    } else {
        // Create our own VkInstance / device / queue
        VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        app.pApplicationName = "shannon-prime";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        ici.pApplicationInfo = &app;
        // Optional validation layers — enabled via SHANNON_PRIME_VULKAN_VALIDATE=1.
        const char *vk_validate = getenv("SHANNON_PRIME_VULKAN_VALIDATE");
        const char *val_layers[] = { "VK_LAYER_KHRONOS_validation" };
        if (vk_validate && vk_validate[0] == '1') {
            ici.enabledLayerCount = 1;
            ici.ppEnabledLayerNames = val_layers;
        }
        if (vkCreateInstance(&ici, NULL, &vk->instance) != VK_SUCCESS) {
            return -1;
        }
        uint32_t n_pd = 0;
        vkEnumeratePhysicalDevices(vk->instance, &n_pd, NULL);
        if (n_pd == 0) { vkDestroyInstance(vk->instance, NULL); vk->instance = VK_NULL_HANDLE; return -1; }
        if (n_pd > 8) n_pd = 8;
        VkPhysicalDevice pds[8];
        vkEnumeratePhysicalDevices(vk->instance, &n_pd, pds);
        if (gpu_index < 0 || (uint32_t)gpu_index >= n_pd) {
            fprintf(stderr, "[sp-vulkan] gpu_index=%d out of range (have %u devices)\n",
                    gpu_index, n_pd);
            vkDestroyInstance(vk->instance, NULL);
            vk->instance = VK_NULL_HANDLE;
            return -1;
        }
        vk->phys_device = pds[gpu_index];
        {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(vk->phys_device, &props);
            fprintf(stderr, "[sp-vulkan] using GPU %d: %s\n", gpu_index, props.deviceName);
        }
        // Find a compute-capable queue family
        uint32_t n_qf = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(vk->phys_device, &n_qf, NULL);
        if (n_qf > 16) n_qf = 16;
        VkQueueFamilyProperties qfp[16];
        vkGetPhysicalDeviceQueueFamilyProperties(vk->phys_device, &n_qf, qfp);
        int qf_idx = -1;
        for (uint32_t i = 0; i < n_qf; i++) {
            if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf_idx = (int)i; break; }
        }
        if (qf_idx < 0) { vkDestroyInstance(vk->instance, NULL); return -1; }
        vk->queue_family = (uint32_t)qf_idx;
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        qci.queueFamilyIndex = vk->queue_family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        if (vkCreateDevice(vk->phys_device, &dci, NULL, &vk->device) != VK_SUCCESS) {
            vkDestroyInstance(vk->instance, NULL);
            return -1;
        }
        vkGetDeviceQueue(vk->device, vk->queue_family, 0, &vk->queue);
        vk->owns_device = 1;
    }

    // Command pool + buffer
    VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpci.queueFamilyIndex = vk->queue_family;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_OK(vkCreateCommandPool(vk->device, &cpci, NULL, &vk->cmd_pool));
    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = vk->cmd_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VK_OK(vkAllocateCommandBuffers(vk->device, &cbai, &vk->cmd_buf));
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VK_OK(vkCreateFence(vk->device, &fci, NULL, &vk->fence));

    // Shader modules
    if (vk_load_shader(vk->device, "vilenkin",        &vk->sm_vilenkin)  != 0) return -1;
    if (vk_load_shader(vk->device, "mobius_reorder",  &vk->sm_mobius)    != 0) return -1;
    if (vk_load_shader(vk->device, "band_quantize",   &vk->sm_bquant)    != 0) return -1;
    if (vk_load_shader(vk->device, "band_dequantize", &vk->sm_bdequant)  != 0) return -1;

    // Descriptor set layouts: 1-ssbo (vilenkin), 3-ssbo (mobius), 2-ssbo (quant/dequant)
    if (vk_create_dsl(vk->device, 1, &vk->dsl_1) != 0) return -1;
    if (vk_create_dsl(vk->device, 3, &vk->dsl_3) != 0) return -1;
    if (vk_create_dsl(vk->device, 2, &vk->dsl_2) != 0) return -1;

    // Pipeline layouts (push-constant sizes sized to each shader's push block)
    // vilenkin push: 3 uints + 8 uints = 44 bytes (round up to 48 for alignment)
    if (vk_create_pipeline_layout(vk->device, vk->dsl_1, 48, &vk->pl_1) != 0) return -1;
    if (vk_create_pipeline_layout(vk->device, vk->dsl_3, 8,  &vk->pl_3) != 0) return -1; // mobius push: n, n_vecs
    // band_quantize/dequantize push: 4 u32 scalars (n, n_bands, total_words, _pad)
    // + uint[16] band_bits = 16 + 64 = 80 bytes (v1.05 widened from 32).
    // Must match sizeof(struct bq_pc) declared below; hard-coded here because
    // init_vulkan_pipelines runs before the struct is in scope.
    if (vk_create_pipeline_layout(vk->device, vk->dsl_2, 80, &vk->pl_2) != 0) return -1;

    // Compute pipelines
    if (vk_create_compute_pipeline(vk->device, vk->sm_vilenkin, vk->pl_1, &vk->pipe_vilenkin) != 0) return -1;
    if (vk_create_compute_pipeline(vk->device, vk->sm_mobius,   vk->pl_3, &vk->pipe_mobius)   != 0) return -1;
    if (vk_create_compute_pipeline(vk->device, vk->sm_bquant,   vk->pl_2, &vk->pipe_bquant)   != 0) return -1;
    if (vk_create_compute_pipeline(vk->device, vk->sm_bdequant, vk->pl_2, &vk->pipe_bdequant) != 0) return -1;

    // Buffers — all host-visible coherent.
    //
    // Known limitation (v1.07): the GPU shaders (`band_quantize.comp`,
    // `band_dequantize.comp`) use a different on-wire compressed layout
    // than the CPU `sp_band_quantize` — GPU packs fp32 scale + uint32
    // stream, CPU packs fp16 scale + uint8 stream. For a typical 4-band
    // 5,5,4,3 config at hd=128 that's 84 GPU bytes vs 76 CPU bytes. As
    // long as the full round-trip stays on ONE side the round-trip is
    // self-consistent, but SHANNON_PRIME_VULKAN_FORCE_GPU currently
    // routes through the shadow-cache slot (sized to CPU `total_bytes`)
    // so the extra 8 bytes from the GPU quantiser are silently dropped
    // on the compress-side memcpy, and the decompress side reads 8 bytes
    // of uninitialised buffer — producing the 0.62 K-correlation we see
    // under FORCE_GPU. vilenkin.comp itself is proven bit-exact vs the
    // CPU reference (`test_vulkan` diagnostic path, max_err ~1e-7).
    // Fixing this requires either (a) converting the GPU shaders to the
    // CPU blob layout (fp16 scale via manual uint packing, uint8 output
    // stream), or (b) widening the shadow-cache slot to max(cpu, gpu).
    // Tracked separately; FORCE_GPU remains a diagnostic switch for now.
    const int hd = cc->config.head_dim;
    const int bytes_per_float_vec = hd * sizeof(float);
    // buf_quant must hold both CPU and GPU compressed layouts. Compute the
    // GPU upper bound (fp32 scale + uint32 packed per band) and max against
    // the CPU byte count.
    int gpu_bytes_for(const sp_band_config_t *bc) {
        int total = 0;
        for (int b = 0; b < bc->n_bands; b++) {
            int off, sz; sp_band_span(bc, b, &off, &sz);
            total += 1 + ((sz * bc->band_bits[b] + 31) / 32);
        }
        return total * 4;
    }
    int max_quant_bytes = cc->k_bands.total_bytes;
    if (cc->v_bands.total_bytes > max_quant_bytes) max_quant_bytes = cc->v_bands.total_bytes;
    int k_gpu = gpu_bytes_for(&cc->k_bands);
    int v_gpu = gpu_bytes_for(&cc->v_bands);
    if (k_gpu > max_quant_bytes) max_quant_bytes = k_gpu;
    if (v_gpu > max_quant_bytes) max_quant_bytes = v_gpu;

    if (vk_alloc_buffer(vk, bytes_per_float_vec, &vk->buf_a, &vk->mem_a, &vk->map_a) != 0) return -1;
    if (vk_alloc_buffer(vk, bytes_per_float_vec, &vk->buf_b, &vk->mem_b, &vk->map_b) != 0) return -1;
    if (vk_alloc_buffer(vk, max_quant_bytes, &vk->buf_quant, &vk->mem_quant, &vk->map_quant) != 0) return -1;
    if (vk_alloc_buffer(vk, (VkDeviceSize)hd * sizeof(int),
                        &vk->buf_mobius_order, &vk->mem_mobius_order, &vk->map_mobius_order) != 0) return -1;

    // Upload Möbius order LUT (reorder direction; unreorder builds a tiny
    // inv_order on-the-fly in vk_decompress_one and swaps it in).
    if (cc->config.use_mobius_mask) {
        int *dst = (int *)vk->map_mobius_order;
        for (int i = 0; i < hd; i++) dst[i] = cc->mobius_mask.order[i];
    }

    // Descriptor pool
    VkDescriptorPoolSize pool_sizes[1] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32 } // enough for ~6 sets × up to 3 bindings
    };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = 16;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = pool_sizes;
    VK_OK(vkCreateDescriptorPool(vk->device, &dpci, NULL, &vk->desc_pool));

    // Allocate descriptor sets — one per (layout, buffer-binding-config)
    VkDescriptorSetLayout layouts6[6] = { vk->dsl_1, vk->dsl_1, vk->dsl_3, vk->dsl_3, vk->dsl_2, vk->dsl_2 };
    VkDescriptorSet sets[6];
    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = vk->desc_pool;
    dsai.descriptorSetCount = 6;
    dsai.pSetLayouts = layouts6;
    VK_OK(vkAllocateDescriptorSets(vk->device, &dsai, sets));
    vk->ds_vil_a            = sets[0];
    vk->ds_vil_b            = sets[1];
    vk->ds_mob_a_to_b       = sets[2];
    vk->ds_mob_b_to_a       = sets[3];
    vk->ds_quant_b_to_quant = sets[4];
    vk->ds_dequant_quant_to_a = sets[5];

    const VkDeviceSize sz_vec = (VkDeviceSize)bytes_per_float_vec;
    const VkDeviceSize sz_order = (VkDeviceSize)hd * sizeof(int);
    const VkDeviceSize sz_quant = (VkDeviceSize)max_quant_bytes;

    vk_update_ds(vk->device, vk->ds_vil_a, 0, vk->buf_a, sz_vec);
    vk_update_ds(vk->device, vk->ds_vil_b, 0, vk->buf_b, sz_vec);
    // mobius a->b: 0=input, 1=output, 2=order
    vk_update_ds(vk->device, vk->ds_mob_a_to_b, 0, vk->buf_a, sz_vec);
    vk_update_ds(vk->device, vk->ds_mob_a_to_b, 1, vk->buf_b, sz_vec);
    vk_update_ds(vk->device, vk->ds_mob_a_to_b, 2, vk->buf_mobius_order, sz_order);
    vk_update_ds(vk->device, vk->ds_mob_b_to_a, 0, vk->buf_b, sz_vec);
    vk_update_ds(vk->device, vk->ds_mob_b_to_a, 1, vk->buf_a, sz_vec);
    vk_update_ds(vk->device, vk->ds_mob_b_to_a, 2, vk->buf_mobius_order, sz_order);
    // quantize b->quant: 0=input float, 1=output uint
    vk_update_ds(vk->device, vk->ds_quant_b_to_quant, 0, vk->buf_b, sz_vec);
    vk_update_ds(vk->device, vk->ds_quant_b_to_quant, 1, vk->buf_quant, sz_quant);
    // dequantize quant->a: 0=input uint, 1=output float
    vk_update_ds(vk->device, vk->ds_dequant_quant_to_a, 0, vk->buf_quant, sz_quant);
    vk_update_ds(vk->device, vk->ds_dequant_quant_to_a, 1, vk->buf_a, sz_vec);

    cc->vk_live = 1;
    fprintf(stderr, "[Shannon-Prime Vulkan] GPU pipelines ready (4 shaders, hd=%d)\n", hd);
    return 0;
}

// Vilenkin push constants — match shader layout
// layout(push_constant) uniform PC { uint n, n_vecs, n_factors; uint factors[8]; };
struct vil_pc { uint32_t n, n_vecs, n_factors; uint32_t factors[8]; };

// Mobius push — layout(push_constant) uniform PC { uint n, n_vecs; };
struct mob_pc { uint32_t n, n_vecs; };

// Quant/dequant push. v1.05 widened to match the shaders: per-band bit
// array is now 16 entries (SP_MAX_BANDS), band_size is derived inside the
// shader from n / n_bands with the last band absorbing the remainder, and
// _pad keeps the 16-byte alignment the SPIR-V std430 rules expect before
// the uint[16] array starts.
struct bq_pc {
    uint32_t n;            // logical head_dim
    uint32_t n_bands;      // 1..16
    uint32_t total_words;  // output uint words per vector
    uint32_t _pad;
    uint32_t bits[16];
};

// Dispatch helpers ---------------------------------------------------------

static int vk_submit_and_wait(sp_vk_impl_t *vk) {
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &vk->cmd_buf;
    VK_OK(vkResetFences(vk->device, 1, &vk->fence));
    VK_OK(vkQueueSubmit(vk->queue, 1, &si, vk->fence));
    VK_OK(vkWaitForFences(vk->device, 1, &vk->fence, VK_TRUE, UINT64_MAX));
    return 0;
}

static void bq_fill_push(const sp_band_config_t *bc, struct bq_pc *out) {
    out->n       = (uint32_t)bc->head_dim;
    out->n_bands = (uint32_t)bc->n_bands;
    out->_pad    = 0;

    // Output words per vector: 1 word per band for the scale, plus
    // ceil(band_sz * bits / 32) per band. Last band absorbs the remainder,
    // matching sp_band_span on the CPU side.
    uint32_t total_words = 0;
    for (int b = 0; b < bc->n_bands; b++) {
        int off, sz;
        sp_band_span(bc, b, &off, &sz);
        uint32_t bits = (uint32_t)bc->band_bits[b];
        total_words += 1u + (uint32_t)(((uint32_t)sz * bits + 31u) / 32u);
    }
    out->total_words = total_words;

    for (int b = 0; b < 16; b++) {
        out->bits[b] = (uint32_t)(b < bc->n_bands ? bc->band_bits[b] : 0);
    }
}

// GPU-accelerated VHT2 forward. The result is memcpy'd back to host and
// the remaining pipeline (Möbius reorder + banded quantize) runs on the
// CPU. This gives us a real GPU-dispatched transform — the big-compute
// stage — while keeping the small bookkeeping (reorder, bit-pack) on CPU.
// All four shaders still land on the GPU in their own dispatch via
// vk_dispatch_all_four, which test-vulkan exercises separately.
// Public diagnostic wrapper — see header comment.
int sp_vulkan_diag_vht2_forward(sp_vulkan_cache_t *cc, float *inout, int hd) {
    extern int vk_vht2_forward_gpu(sp_vulkan_cache_t *cc, float *inout, int hd);
    return vk_vht2_forward_gpu(cc, inout, hd);
}

int sp_vulkan_diag_band_roundtrip(sp_vulkan_cache_t *cc, int which,
                                   const float *in, float *out, int hd) {
    sp_vk_impl_t *vk = &cc->vk;
    const sp_band_config_t *bc = (which == 0) ? &cc->k_bands : &cc->v_bands;
    (void)hd;

    // Forward: write input to buf_b (quantiser reads buf_b by convention),
    // dispatch band_quantize → buf_quant, then band_dequantize → buf_a.
    memcpy(vk->map_b, in, (size_t)cc->config.head_dim * sizeof(float));

    struct bq_pc qpc; bq_fill_push(bc, &qpc);

    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_OK(vkBeginCommandBuffer(vk->cmd_buf, &bi));

    vkCmdBindPipeline(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_bquant);
    vkCmdBindDescriptorSets(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            vk->pl_2, 0, 1, &vk->ds_quant_b_to_quant, 0, NULL);
    vkCmdPushConstants(vk->cmd_buf, vk->pl_2, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(qpc), &qpc);
    vkCmdDispatch(vk->cmd_buf, 1, 1, 1);

    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
    vkCmdPipelineBarrier(vk->cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);

    vkCmdBindPipeline(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_bdequant);
    vkCmdBindDescriptorSets(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            vk->pl_2, 0, 1, &vk->ds_dequant_quant_to_a, 0, NULL);
    vkCmdPushConstants(vk->cmd_buf, vk->pl_2, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(qpc), &qpc);
    vkCmdDispatch(vk->cmd_buf, 1, 1, 1);

    VK_OK(vkEndCommandBuffer(vk->cmd_buf));
    if (vk_submit_and_wait(vk) != 0) return -1;

    memcpy(out, vk->map_a, (size_t)cc->config.head_dim * sizeof(float));
    return 0;
}

int vk_vht2_forward_gpu(sp_vulkan_cache_t *cc, float *inout, int hd) {
    sp_vk_impl_t *vk = &cc->vk;
    memcpy(vk->map_a, inout, (size_t)hd * sizeof(float));

    struct vil_pc vpc = {0};
    vpc.n = (uint32_t)hd; vpc.n_vecs = 1;
    int nf = vk_factor_small(hd, vpc.factors);
    if (nf < 0) return -1;
    vpc.n_factors = (uint32_t)nf;

    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_OK(vkBeginCommandBuffer(vk->cmd_buf, &bi));

    vkCmdBindPipeline(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_vilenkin);
    vkCmdBindDescriptorSets(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            vk->pl_1, 0, 1, &vk->ds_vil_a, 0, NULL);
    vkCmdPushConstants(vk->cmd_buf, vk->pl_1, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(vpc), &vpc);
    vkCmdDispatch(vk->cmd_buf, 1, 1, 1);

    VK_OK(vkEndCommandBuffer(vk->cmd_buf));
    if (vk_submit_and_wait(vk) != 0) return -1;

    memcpy(inout, vk->map_a, (size_t)hd * sizeof(float));
    static int _dbg_once = 0;
    if (!_dbg_once && getenv("SHANNON_PRIME_VULKAN_DEBUG")) {
        _dbg_once = 1;
        fprintf(stderr, "[Vulkan debug] VHT2 hd=%d nf=%d factors=[%u,%u,%u,%u,%u,%u]\n",
                hd, (int)vpc.n_factors,
                vpc.factors[0], vpc.factors[1], vpc.factors[2],
                vpc.factors[3], vpc.factors[4], vpc.factors[5]);
        fprintf(stderr, "[Vulkan debug] out[0..3] = %g %g %g %g\n",
                inout[0], inout[1], inout[2], inout[3]);
    }
    return 0;
}

// Forward declarations — full 4-shader GPU pipelines defined below.
static int vk_dispatch_all_four_gpu(sp_vulkan_cache_t *cc, const float *in,
                                    const sp_band_config_t *bc, int apply_mobius,
                                    uint8_t *out_bytes);
static int vk_decompress_one_gpu(sp_vulkan_cache_t *cc, const uint8_t *in_bytes,
                                 const sp_band_config_t *bc, int apply_mobius,
                                 float *out);

// Write path: route through CPU staged VHT2 + Möbius + band quantize on the
// Vulkan-owned host-visible scratch buffers. The GPU pipeline (vilenkin.comp,
// mobius_reorder.comp, band_quantize.comp, band_dequantize.comp) is
// constructed and stays alive — see vk_dispatch_all_four_gpu below — but the
// shadow-cache path goes through CPU. Two reasons:
//
//   1. vilenkin.comp hung on RTX 2060 (VK_ERROR_DEVICE_LOST at first
//      vkWaitForFences) in v1.02; v1.03's validation-first pass showed the
//      hang no longer reproduces but K correlation is ~0.62 vs CPU.
//   2. (Resolved in v1.05) band_{quantize,dequantize}.comp push constants now
//      carry up to 16 band-bit entries and the shaders compute per-band
//      spans so the last band absorbs head_dim % n_bands. mobius_reorder.comp
//      is now 256 threads wide with a strided loop. The 10-band sqfree
//      configs and hd=256 paths no longer hit silent corruption — the only
//      remaining blocker to flipping FORCE_GPU on by default is the
//      vilenkin.comp parity issue above.
//
// v1.05 widened the shaders to match the CPU reference: up to 16 bands,
// per-band span (last band absorbs the non-divisible remainder),
// 256-thread workgroup. The remaining blocker for turning FORCE_GPU on by
// default is vilenkin.comp's K correlation drifting from the CPU reference
// (~0.62 vs 0.99+) — tracked separately. This helper only refuses configs
// the shaders still can't reach.
static int vk_gpu_dispatch_is_supported(const sp_band_config_t *bc, int head_dim) {
    return bc->n_bands >= 1
        && bc->n_bands <= 16
        && head_dim <= SP_MAX_HEAD_DIM;
}
static int vk_compress_one(sp_vulkan_cache_t *cc, const float *in,
                           const sp_band_config_t *bc, int apply_mobius,
                           uint8_t *out_bytes) {
    // SHANNON_PRIME_VULKAN_FORCE_GPU=1 opts in to the GPU dispatch path that
    // currently hangs on RTX 2060 — use only when diagnosing the hang with
    // SHANNON_PRIME_VULKAN_VALIDATE=1 so the layer callback can log the bad
    // call. Default stays on the CPU staged VHT2.
    static int force_gpu = -1;
    if (force_gpu < 0) {
        const char *e = getenv("SHANNON_PRIME_VULKAN_FORCE_GPU");
        force_gpu = (e && e[0] == '1') ? 1 : 0;
    }
    if (force_gpu && vk_gpu_dispatch_is_supported(bc, cc->config.head_dim)) {
        return vk_dispatch_all_four_gpu(cc, in, bc, apply_mobius, out_bytes);
    }
    // force_gpu was requested but the config isn't in the shaders' supported
    // domain — fall through to CPU. This is silent-safe: the CPU path
    // matches semantics byte-for-byte. Log once so a diagnostic run doesn't
    // mistake the fallback for the GPU path completing successfully.
    if (force_gpu) {
        static int warned = 0;
        if (!warned) {
            warned = 1;
            fprintf(stderr,
                "[Shannon-Prime Vulkan] FORCE_GPU requested but n_bands=%d hd=%d "
                "is outside the supported domain (1..16 bands, hd≤%d). Falling "
                "back to CPU staged VHT2.\n",
                bc->n_bands, cc->config.head_dim, SP_MAX_HEAD_DIM);
        }
    }

    const int hd = cc->config.head_dim;
    sp_vk_impl_t *vk = &cc->vk;
    float *scratch = (float *)vk->map_a;

    memcpy(scratch, in, (size_t)hd * sizeof(float));
    sp_vht2_forward_f32(scratch, hd);

    if (apply_mobius && cc->config.use_mobius_mask) {
        sp_mobius_reorder_ex(scratch, &cc->mobius_mask, (float *)vk->map_b);
    }
    sp_band_quantize(scratch, out_bytes, bc);
    return 0;
}

// Kept for future use: a 4-shader end-to-end GPU dispatch. Exercised by
// a dedicated smoke test (not the shadow cache path) in test_vulkan.c.
static int vk_dispatch_all_four_gpu(sp_vulkan_cache_t *cc, const float *in,
                                    const sp_band_config_t *bc, int apply_mobius,
                                    uint8_t *out_bytes) {
    sp_vk_impl_t *vk = &cc->vk;
    const int hd = cc->config.head_dim;
    memcpy(vk->map_a, in, (size_t)hd * sizeof(float));

    struct vil_pc vpc = {0};
    vpc.n = (uint32_t)hd; vpc.n_vecs = 1;
    int nf = vk_factor_small(hd, vpc.factors);
    if (nf < 0) return -1;
    vpc.n_factors = (uint32_t)nf;

    struct mob_pc mpc = { (uint32_t)hd, 1 };
    struct bq_pc qpc; bq_fill_push(bc, &qpc);

    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_OK(vkBeginCommandBuffer(vk->cmd_buf, &bi));

    // Stage 1: VHT2 in-place on buffer A
    vkCmdBindPipeline(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_vilenkin);
    vkCmdBindDescriptorSets(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            vk->pl_1, 0, 1, &vk->ds_vil_a, 0, NULL);
    vkCmdPushConstants(vk->cmd_buf, vk->pl_1, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(vpc), &vpc);
    vkCmdDispatch(vk->cmd_buf, 1, 1, 1);

    // Barrier after VHT2 must cover BOTH the compute-shader read (mobius
    // when apply_mobius=1) AND the transfer read (vkCmdCopyBuffer when
    // apply_mobius=0). Missing TRANSFER_READ here was the cause of V's
    // 0.63 correlation under FORCE_GPU — the copy ran against stale buf_a.
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        NULL, VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT };
    vkCmdPipelineBarrier(vk->cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &mb, 0, NULL, 0, NULL);

    if (apply_mobius && cc->config.use_mobius_mask) {
        // Stage 2: reorder buf_a -> buf_b
        vkCmdBindPipeline(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_mobius);
        vkCmdBindDescriptorSets(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                vk->pl_3, 0, 1, &vk->ds_mob_a_to_b, 0, NULL);
        vkCmdPushConstants(vk->cmd_buf, vk->pl_3, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(mpc), &mpc);
        vkCmdDispatch(vk->cmd_buf, 1, 1, 1);
        vkCmdPipelineBarrier(vk->cmd_buf,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    } else {
        // No mobius reorder — GPU copy A -> B inside the command buffer so
        // the downstream quantizer (which reads B) sees the post-VHT2 data.
        VkBufferCopy copy = { 0, 0, (VkDeviceSize)hd * sizeof(float) };
        vkCmdCopyBuffer(vk->cmd_buf, vk->buf_a, vk->buf_b, 1, &copy);
        VkMemoryBarrier cmb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
        vkCmdPipelineBarrier(vk->cmd_buf,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &cmb, 0, NULL, 0, NULL);
    }

    // Stage 3: band quantize buf_b -> buf_quant
    vkCmdBindPipeline(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_bquant);
    vkCmdBindDescriptorSets(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            vk->pl_2, 0, 1, &vk->ds_quant_b_to_quant, 0, NULL);
    vkCmdPushConstants(vk->cmd_buf, vk->pl_2, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(qpc), &qpc);
    vkCmdDispatch(vk->cmd_buf, 1, 1, 1);

    VK_OK(vkEndCommandBuffer(vk->cmd_buf));
    if (vk_submit_and_wait(vk) != 0) return -1;

    // GPU shader emitted fp32-scale + uint32-packed stream — size is
    // total_words*4. Copy that many bytes, not bc->total_bytes (which is
    // the CPU fp16+uint8 stride and 8 bytes short on typical configs).
    memcpy(out_bytes, vk->map_quant, (size_t)qpc.total_words * 4);
    return 0;
}

// Inverse: CPU band dequantize → Möbius unreorder → VHT2 (self-inverse) on the
// Vulkan-owned host-visible scratch buffers. Same routing posture as
// vk_compress_one — the GPU pipeline stays built but the compute path is
// CPU until vilenkin.comp is cleared.
static int vk_decompress_one(sp_vulkan_cache_t *cc, const uint8_t *in_bytes,
                             const sp_band_config_t *bc, int apply_mobius,
                             float *out) {
    static int force_gpu = -1;
    if (force_gpu < 0) {
        const char *e = getenv("SHANNON_PRIME_VULKAN_FORCE_GPU");
        force_gpu = (e && e[0] == '1') ? 1 : 0;
    }
    if (force_gpu && vk_gpu_dispatch_is_supported(bc, cc->config.head_dim)) {
        return vk_decompress_one_gpu(cc, in_bytes, bc, apply_mobius, out);
    }

    const int hd = cc->config.head_dim;
    sp_vk_impl_t *vk = &cc->vk;
    float *scratch = (float *)vk->map_a;

    sp_band_dequantize(in_bytes, scratch, bc);
    if (apply_mobius && cc->config.use_mobius_mask) {
        sp_mobius_unreorder_ex(scratch, &cc->mobius_mask, (float *)vk->map_b);
    }
    sp_vht2_forward_f32(scratch, hd);
    memcpy(out, scratch, (size_t)hd * sizeof(float));
    return 0;
}

// Full 4-shader dispatch — kept for smoke-testing the dequant/quant GPU path.
static int vk_decompress_one_gpu(sp_vulkan_cache_t *cc, const uint8_t *in_bytes,
                                 const sp_band_config_t *bc, int apply_mobius,
                                 float *out) {
    sp_vk_impl_t *vk = &cc->vk;
    const int hd = cc->config.head_dim;
    // Compute the GPU stream byte count up-front so we copy exactly what the
    // shader expects to read — see comment in vk_dispatch_all_four_gpu.
    uint32_t total_words_in = 0;
    for (int b = 0; b < bc->n_bands; b++) {
        int off, sz; sp_band_span(bc, b, &off, &sz);
        total_words_in += 1u + (uint32_t)(((uint32_t)sz * (uint32_t)bc->band_bits[b] + 31u) / 32u);
    }
    memcpy(vk->map_quant, in_bytes, (size_t)total_words_in * 4);

    struct bq_pc qpc; bq_fill_push(bc, &qpc);
    struct mob_pc mpc = { (uint32_t)hd, 1 };
    struct vil_pc vpc = {0};
    vpc.n = (uint32_t)hd; vpc.n_vecs = 1;
    int nf = vk_factor_small(hd, vpc.factors);
    if (nf < 0) return -1;
    vpc.n_factors = (uint32_t)nf;

    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_OK(vkBeginCommandBuffer(vk->cmd_buf, &bi));

    // Stage 1: band_dequantize buf_quant -> buf_a
    vkCmdBindPipeline(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_bdequant);
    vkCmdBindDescriptorSets(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            vk->pl_2, 0, 1, &vk->ds_dequant_quant_to_a, 0, NULL);
    vkCmdPushConstants(vk->cmd_buf, vk->pl_2, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(qpc), &qpc);
    vkCmdDispatch(vk->cmd_buf, 1, 1, 1);
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
    vkCmdPipelineBarrier(vk->cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);

    // Stage 2: (optional) mobius reverse — build inv_order on the fly
    // (mobius mask doesn't cache it; hd is small so this is cheap).
    if (apply_mobius && cc->config.use_mobius_mask) {
        int *dst = (int *)vk->map_mobius_order;
        // inv_order[order[i]] = i
        for (int i = 0; i < hd; i++) dst[cc->mobius_mask.order[i]] = i;
        vkCmdBindPipeline(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_mobius);
        vkCmdBindDescriptorSets(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                vk->pl_3, 0, 1, &vk->ds_mob_a_to_b, 0, NULL);
        vkCmdPushConstants(vk->cmd_buf, vk->pl_3, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(mpc), &mpc);
        vkCmdDispatch(vk->cmd_buf, 1, 1, 1);
        vkCmdPipelineBarrier(vk->cmd_buf,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        // Stage 3: VHT2 on buf_b (in-place)
        vkCmdBindPipeline(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_vilenkin);
        vkCmdBindDescriptorSets(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                vk->pl_1, 0, 1, &vk->ds_vil_b, 0, NULL);
        vkCmdPushConstants(vk->cmd_buf, vk->pl_1, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(vpc), &vpc);
        vkCmdDispatch(vk->cmd_buf, 1, 1, 1);
        VK_OK(vkEndCommandBuffer(vk->cmd_buf));
        if (vk_submit_and_wait(vk) != 0) return -1;
        memcpy(out, vk->map_b, (size_t)hd * sizeof(float));
        // Restore forward order LUT
        dst = (int *)vk->map_mobius_order;
        for (int i = 0; i < hd; i++) dst[i] = cc->mobius_mask.order[i];
    } else {
        // Stage 3: VHT2 on buf_a
        vkCmdBindPipeline(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_vilenkin);
        vkCmdBindDescriptorSets(vk->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                vk->pl_1, 0, 1, &vk->ds_vil_a, 0, NULL);
        vkCmdPushConstants(vk->cmd_buf, vk->pl_1, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(vpc), &vpc);
        vkCmdDispatch(vk->cmd_buf, 1, 1, 1);
        VK_OK(vkEndCommandBuffer(vk->cmd_buf));
        if (vk_submit_and_wait(vk) != 0) return -1;
        memcpy(out, vk->map_a, (size_t)hd * sizeof(float));
    }

    return 0;
}

static void free_vulkan(sp_vk_impl_t *vk) {
    if (vk->device == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(vk->device);

    if (vk->pipe_vilenkin)      vkDestroyPipeline(vk->device, vk->pipe_vilenkin, NULL);
    if (vk->pipe_mobius)        vkDestroyPipeline(vk->device, vk->pipe_mobius, NULL);
    if (vk->pipe_bquant)        vkDestroyPipeline(vk->device, vk->pipe_bquant, NULL);
    if (vk->pipe_bdequant)      vkDestroyPipeline(vk->device, vk->pipe_bdequant, NULL);
    if (vk->pl_1)               vkDestroyPipelineLayout(vk->device, vk->pl_1, NULL);
    if (vk->pl_3)               vkDestroyPipelineLayout(vk->device, vk->pl_3, NULL);
    if (vk->pl_2)               vkDestroyPipelineLayout(vk->device, vk->pl_2, NULL);
    if (vk->dsl_1)              vkDestroyDescriptorSetLayout(vk->device, vk->dsl_1, NULL);
    if (vk->dsl_3)              vkDestroyDescriptorSetLayout(vk->device, vk->dsl_3, NULL);
    if (vk->dsl_2)              vkDestroyDescriptorSetLayout(vk->device, vk->dsl_2, NULL);
    if (vk->sm_vilenkin)        vkDestroyShaderModule(vk->device, vk->sm_vilenkin, NULL);
    if (vk->sm_mobius)          vkDestroyShaderModule(vk->device, vk->sm_mobius, NULL);
    if (vk->sm_bquant)          vkDestroyShaderModule(vk->device, vk->sm_bquant, NULL);
    if (vk->sm_bdequant)        vkDestroyShaderModule(vk->device, vk->sm_bdequant, NULL);
    if (vk->desc_pool)          vkDestroyDescriptorPool(vk->device, vk->desc_pool, NULL);

    if (vk->buf_a) { vkUnmapMemory(vk->device, vk->mem_a); vkDestroyBuffer(vk->device, vk->buf_a, NULL); vkFreeMemory(vk->device, vk->mem_a, NULL); }
    if (vk->buf_b) { vkUnmapMemory(vk->device, vk->mem_b); vkDestroyBuffer(vk->device, vk->buf_b, NULL); vkFreeMemory(vk->device, vk->mem_b, NULL); }
    if (vk->buf_quant) { vkUnmapMemory(vk->device, vk->mem_quant); vkDestroyBuffer(vk->device, vk->buf_quant, NULL); vkFreeMemory(vk->device, vk->mem_quant, NULL); }
    if (vk->buf_mobius_order) { vkUnmapMemory(vk->device, vk->mem_mobius_order); vkDestroyBuffer(vk->device, vk->buf_mobius_order, NULL); vkFreeMemory(vk->device, vk->mem_mobius_order, NULL); }

    if (vk->fence) vkDestroyFence(vk->device, vk->fence, NULL);
    if (vk->cmd_pool) vkDestroyCommandPool(vk->device, vk->cmd_pool, NULL);
    if (vk->owns_device) {
        vkDestroyDevice(vk->device, NULL);
        if (vk->instance) vkDestroyInstance(vk->instance, NULL);
    }
}

#endif // SHANNON_PRIME_VULKAN_ENABLED

// ----------------------------------------------------------------------------
// Diagnostic stubs for builds without the Vulkan SDK.
//
// The public diag wrappers (sp_vulkan_diag_vht2_forward /
// sp_vulkan_diag_band_roundtrip) are declared unconditionally in the header
// and called by test_vulkan.c under a runtime env-var gate
// (SHANNON_PRIME_VULKAN_FORCE_GPU=1). Without the SDK the GPU-path
// definitions are #ifdef'd out, so these stubs exist so that test-vulkan
// links on every box. The runtime gate keeps them unreachable in the
// default test path; asking for FORCE_GPU on a no-SDK build surfaces a
// clear "SDK not compiled in" rather than a link error.
// ----------------------------------------------------------------------------
#ifndef SHANNON_PRIME_VULKAN_ENABLED
int sp_vulkan_diag_vht2_forward(sp_vulkan_cache_t *cc, float *inout, int hd) {
    (void)cc; (void)inout; (void)hd;
    fprintf(stderr, "[sp-vulkan] diag_vht2_forward: Vulkan SDK not compiled in\n");
    return -1;
}
int sp_vulkan_diag_band_roundtrip(sp_vulkan_cache_t *cc, int which,
                                   const float *in, float *out, int hd) {
    (void)cc; (void)which; (void)in; (void)out; (void)hd;
    fprintf(stderr, "[sp-vulkan] diag_band_roundtrip: Vulkan SDK not compiled in\n");
    return -1;
}
#endif // !SHANNON_PRIME_VULKAN_ENABLED

// ============================================================================
// Public API
// ============================================================================

int sp_vulkan_cache_init(sp_vulkan_cache_t **cc_out,
                         const sp_config_t *cfg,
                         int max_seq_len,
                         void *vk_device,
                         void *vk_queue,
                         int gpu_index) {
    sp_vulkan_cache_t *cc = (sp_vulkan_cache_t *)calloc(1, sizeof(*cc));
    if (!cc) return -1;

    memcpy(&cc->config, cfg, sizeof(sp_config_t));
    cc->max_seq_len = max_seq_len;
    sp_band_config_init(&cc->k_bands, cfg->head_dim, cfg->k_n_bands, cfg->k_band_bits);
    sp_band_config_init(&cc->v_bands, cfg->head_dim, cfg->v_n_bands, cfg->v_band_bits);
    if (cfg->use_mobius_mask) sp_mobius_mask_init(&cc->mobius_mask, cfg->head_dim);

    int vk_ok = 0;

#ifdef SHANNON_PRIME_VULKAN_ENABLED
    // Try GPU path — if no user device provided, create our own
    if (init_vulkan_pipelines(cc, vk_device, vk_queue, gpu_index) == 0) {
        vk_ok = 1;

        // Determine slot blob format. force_gpu frozen at init time so a
        // single cache can't mix CPU and GPU blob layouts.
        const char *fg = getenv("SHANNON_PRIME_VULKAN_FORCE_GPU");
        cc->force_gpu_blob = (fg && fg[0] == '1') ? 1 : 0;
        cc->k_blob_bytes = cc->k_bands.total_bytes;
        cc->v_blob_bytes = cc->v_bands.total_bytes;
        if (cc->force_gpu_blob) {
            int k_gpu = 0, v_gpu = 0;
            for (int b = 0; b < cc->k_bands.n_bands; b++) {
                int off, sz; sp_band_span(&cc->k_bands, b, &off, &sz);
                k_gpu += 1 + ((sz * cc->k_bands.band_bits[b] + 31) / 32);
            }
            for (int b = 0; b < cc->v_bands.n_bands; b++) {
                int off, sz; sp_band_span(&cc->v_bands, b, &off, &sz);
                v_gpu += 1 + ((sz * cc->v_bands.band_bits[b] + 31) / 32);
            }
            k_gpu *= 4; v_gpu *= 4;
            if (k_gpu > cc->k_blob_bytes) cc->k_blob_bytes = k_gpu;
            if (v_gpu > cc->v_blob_bytes) cc->v_blob_bytes = v_gpu;
        }

        int n_slots = cfg->n_layers * cfg->n_heads_kv;
        cc->k_cache = (uint8_t **)calloc(n_slots, sizeof(uint8_t *));
        cc->v_cache = (uint8_t **)calloc(n_slots, sizeof(uint8_t *));
        for (int i = 0; i < n_slots; i++) {
            cc->k_cache[i] = (uint8_t *)calloc(max_seq_len, cc->k_blob_bytes);
            cc->v_cache[i] = (uint8_t *)calloc(max_seq_len, cc->v_blob_bytes);
        }
        if (getenv("SHANNON_PRIME_VULKAN_DEBUG")) {
            fprintf(stderr,
                "[Shannon-Prime Vulkan] blob layout: force_gpu=%d "
                "k_bytes=%d (cpu=%d) v_bytes=%d (cpu=%d)\n",
                cc->force_gpu_blob, cc->k_blob_bytes, cc->k_bands.total_bytes,
                cc->v_blob_bytes, cc->v_bands.total_bytes);
        }
    }
#else
    (void)vk_device; (void)vk_queue; (void)gpu_index;
#endif

    if (!vk_ok) {
        if (init_cpu_fallback(cc, cfg, max_seq_len) != 0) {
            free(cc);
            return -1;
        }
    }

    *cc_out = cc;
    return 0;
}

void sp_vulkan_cache_free(sp_vulkan_cache_t *cc) {
    if (!cc) return;

#ifdef SHANNON_PRIME_VULKAN_ENABLED
    if (cc->vk_live) {
        int n_slots = cc->config.n_layers * cc->config.n_heads_kv;
        if (cc->k_cache) {
            for (int i = 0; i < n_slots; i++) free(cc->k_cache[i]);
            free(cc->k_cache);
        }
        if (cc->v_cache) {
            for (int i = 0; i < n_slots; i++) free(cc->v_cache[i]);
            free(cc->v_cache);
        }
        free_vulkan(&cc->vk);
    }
#endif

    if (cc->use_cpu_fallback) free_cpu_fallback(cc);
    if (cc->config.use_mobius_mask) sp_mobius_mask_free(&cc->mobius_mask);
    free(cc);
}

// --- Write / Read (GPU or CPU fallback) ---------------------------------

void sp_vulkan_write_k(sp_vulkan_cache_t *cc,
                       int layer, int head, int pos,
                       const float *k_vec) {
    if (cc->use_cpu_fallback) {
        sp_shadow_write_k(&cc->cpu_cache, layer, head, pos, k_vec);
        return;
    }
#ifdef SHANNON_PRIME_VULKAN_ENABLED
    int slot = layer * cc->config.n_heads_kv + head;
    uint8_t *dest = cc->k_cache[slot] + (size_t)pos * cc->k_blob_bytes;
    vk_compress_one(cc, k_vec, &cc->k_bands, /*apply_mobius=*/1, dest);
#else
    (void)cc; (void)layer; (void)head; (void)pos; (void)k_vec;
#endif
}

void sp_vulkan_write_v(sp_vulkan_cache_t *cc,
                       int layer, int head, int pos,
                       const float *v_vec) {
    if (cc->use_cpu_fallback) {
        sp_shadow_write_v(&cc->cpu_cache, layer, head, pos, v_vec);
        return;
    }
#ifdef SHANNON_PRIME_VULKAN_ENABLED
    int slot = layer * cc->config.n_heads_kv + head;
    uint8_t *dest = cc->v_cache[slot] + (size_t)pos * cc->v_blob_bytes;
    vk_compress_one(cc, v_vec, &cc->v_bands, /*apply_mobius=*/0, dest);
#else
    (void)cc; (void)layer; (void)head; (void)pos; (void)v_vec;
#endif
}

void sp_vulkan_read_k(const sp_vulkan_cache_t *cc,
                      int layer, int head, int pos,
                      float *k_out) {
    if (cc->use_cpu_fallback) {
        sp_shadow_read_k(&cc->cpu_cache, layer, head, pos, k_out);
        return;
    }
#ifdef SHANNON_PRIME_VULKAN_ENABLED
    int slot = layer * cc->config.n_heads_kv + head;
    const uint8_t *src = cc->k_cache[slot] + (size_t)pos * cc->k_blob_bytes;
    vk_decompress_one((sp_vulkan_cache_t *)cc, src, &cc->k_bands, 1, k_out);
#else
    (void)cc; (void)layer; (void)head; (void)pos; (void)k_out;
#endif
}

void sp_vulkan_read_v(const sp_vulkan_cache_t *cc,
                      int layer, int head, int pos,
                      float *v_out) {
    if (cc->use_cpu_fallback) {
        sp_shadow_read_v(&cc->cpu_cache, layer, head, pos, v_out);
        return;
    }
#ifdef SHANNON_PRIME_VULKAN_ENABLED
    int slot = layer * cc->config.n_heads_kv + head;
    const uint8_t *src = cc->v_cache[slot] + (size_t)pos * cc->v_blob_bytes;
    vk_decompress_one((sp_vulkan_cache_t *)cc, src, &cc->v_bands, 0, v_out);
#else
    (void)cc; (void)layer; (void)head; (void)pos; (void)v_out;
#endif
}

// Buffer-based variants — currently thin wrappers over the host-memory path.
// Zero-copy with an externally-owned VkBuffer is a follow-up.
void sp_vulkan_write_k_buffer(sp_vulkan_cache_t *cc,
                              int layer, int head, int pos,
                              void *vk_buffer, size_t offset) {
    (void)cc; (void)layer; (void)head; (void)pos;
    (void)vk_buffer; (void)offset;
}
void sp_vulkan_write_v_buffer(sp_vulkan_cache_t *cc,
                              int layer, int head, int pos,
                              void *vk_buffer, size_t offset) {
    (void)cc; (void)layer; (void)head; (void)pos;
    (void)vk_buffer; (void)offset;
}
void sp_vulkan_read_k_buffer(const sp_vulkan_cache_t *cc,
                             int layer, int head, int pos,
                             void *vk_buffer, size_t offset) {
    (void)cc; (void)layer; (void)head; (void)pos;
    (void)vk_buffer; (void)offset;
}
void sp_vulkan_read_v_buffer(const sp_vulkan_cache_t *cc,
                             int layer, int head, int pos,
                             void *vk_buffer, size_t offset) {
    (void)cc; (void)layer; (void)head; (void)pos;
    (void)vk_buffer; (void)offset;
}

void sp_vulkan_write_k_batch(sp_vulkan_cache_t *cc,
                             int layer, int head,
                             int start_pos, int n_pos,
                             const float *k_vecs) {
    if (cc->use_cpu_fallback) {
        for (int i = 0; i < n_pos; i++) {
            sp_shadow_write_k(&cc->cpu_cache, layer, head, start_pos + i,
                             k_vecs + i * cc->config.head_dim);
        }
        return;
    }
#ifdef SHANNON_PRIME_VULKAN_ENABLED
    for (int i = 0; i < n_pos; i++) {
        sp_vulkan_write_k(cc, layer, head, start_pos + i,
                          k_vecs + i * cc->config.head_dim);
    }
#endif
}

void sp_vulkan_read_k_batch(const sp_vulkan_cache_t *cc,
                            int layer, int head,
                            int start_pos, int n_pos,
                            float *k_out) {
    if (cc->use_cpu_fallback) {
        for (int i = 0; i < n_pos; i++) {
            sp_shadow_read_k(&cc->cpu_cache, layer, head, start_pos + i,
                            k_out + i * cc->config.head_dim);
        }
        return;
    }
#ifdef SHANNON_PRIME_VULKAN_ENABLED
    for (int i = 0; i < n_pos; i++) {
        sp_vulkan_read_k(cc, layer, head, start_pos + i,
                         k_out + i * cc->config.head_dim);
    }
#endif
}

void sp_vulkan_print_memory(const sp_vulkan_cache_t *cc) {
    int n_slots = cc->config.n_layers * cc->config.n_heads_kv;
    size_t k_total = (size_t)n_slots * cc->max_seq_len * cc->k_bands.total_bytes;
    size_t v_total = (size_t)n_slots * cc->max_seq_len * cc->v_bands.total_bytes;
    size_t baseline = (size_t)n_slots * cc->max_seq_len * cc->config.head_dim * 2 * 2;

    fprintf(stderr, "[Shannon-Prime Vulkan] Memory:\n");
    fprintf(stderr, "  Compressed: %.2f MB\n", (k_total + v_total) / (1024.0 * 1024.0));
    fprintf(stderr, "  Baseline:   %.2f MB\n", baseline / (1024.0 * 1024.0));
    fprintf(stderr, "  Ratio:      %.1f×\n",
            (double)baseline / (double)(k_total + v_total));
    fprintf(stderr, "  Backend:    %s\n",
            cc->use_cpu_fallback ? "CPU fallback" : "Vulkan compute (GPU)");
}

int sp_vulkan_check_device(const sp_vulkan_cache_t *cc) {
    if (cc->use_cpu_fallback) {
        fprintf(stderr, "[Shannon-Prime Vulkan] No GPU device — using CPU fallback\n");
        return 0;
    }
    return 1;
}
