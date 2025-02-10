#include <gst/gst.h>
#include <cuda_runtime.h>
#include "compressionpart.h"

// Define struct for GstFdctCompress
struct _GstFdctCompress {
    GstElement parent;
    float threshold;  // Dynamic threshold property
};

// Define GStreamer element type
G_DEFINE_TYPE(GstFdctCompress, gst_fdct_compress, GST_TYPE_ELEMENT)

// Define static pad templates
static GstStaticPadTemplate src_pad_template =
    GST_STATIC_PAD_TEMPLATE("src",
                            GST_PAD_SRC,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS("video/x-raw, format=BGR"));

static GstStaticPadTemplate sink_pad_template =
    GST_STATIC_PAD_TEMPLATE("sink",
                            GST_PAD_SINK,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS("application/x-rtp, media=video"));

// Function to encode a frame with thresholding
static GstFlowReturn gst_fdct_compress_transform(GstElement *element, GstBuffer *inbuf, GstBuffer *outbuf) {
    GstFdctCompress *self = GST_FDCT_COMPRESS(element);
    GstMapInfo in_map, out_map;
    float *d_input;
    size_t size;

    // Set threshold inside the code (example: modifying dynamically)
    self->threshold = 0.75;  // You can modify this value in code as needed

    // Map input buffer
    if (!gst_buffer_map(inbuf, &in_map, GST_MAP_READ)) {
        g_warning("Failed to map input buffer.");
        return GST_FLOW_ERROR;
    }

    size = in_map.size;

    // Allocate GPU memory
    if (cudaMalloc((void**)&d_input, size) != cudaSuccess) {
        g_warning("CUDA malloc failed.");
        gst_buffer_unmap(inbuf, &in_map);
        return GST_FLOW_ERROR;
    }

    // Copy input frame to device
    if (cudaMemcpy(d_input, in_map.data, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        g_warning("CUDA memcpy to device failed.");
        cudaFree(d_input);
        gst_buffer_unmap(inbuf, &in_map);
        return GST_FLOW_ERROR;
    }

    // Apply threshold dynamically changed inside the code
    apply_threshold_and_compress(d_input, /* width */, /* height */, self->threshold);

    // Allocate output buffer
    outbuf = gst_buffer_new_allocate(NULL, size, NULL);
    if (!gst_buffer_map(outbuf, &out_map, GST_MAP_WRITE)) {
        g_warning("Failed to map output buffer.");
        cudaFree(d_input);
        gst_buffer_unmap(inbuf, &in_map);
        return GST_FLOW_ERROR;
    }

    // Copy back compressed data
    if (cudaMemcpy(out_map.data, d_input, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        g_warning("CUDA memcpy to host failed.");
        gst_buffer_unmap(outbuf, &out_map);
        cudaFree(d_input);
        gst_buffer_unmap(inbuf, &in_map);
        return GST_FLOW_ERROR;
    }

    // Unmap buffers
    gst_buffer_unmap(inbuf, &in_map);
    gst_buffer_unmap(outbuf, &out_map);

    // Free GPU memory
    cudaFree(d_input);

    return GST_FLOW_OK;
}

// Getter function for threshold property
static float gst_fdct_compress_get_threshold(GstFdctCompress *self) {
    return self->threshold;
}

// Setter function for threshold property
static void gst_fdct_compress_set_threshold(GstFdctCompress *self, float threshold) {
    self->threshold = threshold;
}

// Register plugin
GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    fdctcompress,
    "Fast DCT Compression Plugin with Dynamic Thresholding in Code",
    plugin_init,
    "1.0",
    "LGPL",
    "GStreamer",
    "https://gstreamer.freedesktop.org/"
)
