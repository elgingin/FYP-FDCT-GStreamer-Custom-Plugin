#include "decompressionpart.h"
#include <gst/gst.h>
#include <cuda_runtime.h>

// Define struct for GstFdctDecompress
struct _GstFdctDecompress {
    GstElement parent;
};

// Register element type
G_DEFINE_TYPE(GstFdctDecompress, gst_fdct_decompress, GST_TYPE_ELEMENT)

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
                            GST_STATIC_CAPS("video/x-raw, format=BGR"));

// Function to decode a frame
static GstFlowReturn gst_fdct_decompress_transform(GstElement *element, GstBuffer *inbuf, GstBuffer *outbuf) {
    GstMapInfo in_map, out_map;
    float *d_output;
    size_t size;

    // Map input buffer
    if (!gst_buffer_map(inbuf, &in_map, GST_MAP_READ)) {
        g_warning("Failed to map input buffer.");
        return GST_FLOW_ERROR;
    }

    size = in_map.size;

    // Allocate GPU memory
    if (cudaMalloc((void**)&d_output, size) != cudaSuccess) {
        g_warning("CUDA malloc failed.");
        gst_buffer_unmap(inbuf, &in_map);
        return GST_FLOW_ERROR;
    }

    // Copy input frame to device
    if (cudaMemcpy(d_output, in_map.data, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        g_warning("CUDA memcpy to device failed.");
        cudaFree(d_output);
        gst_buffer_unmap(inbuf, &in_map);
        return GST_FLOW_ERROR;
    }

    // Perform decompression
    decompressFrame(d_output, /* width */, /* height */);

    // Allocate output buffer
    outbuf = gst_buffer_new_allocate(NULL, size, NULL);
    if (!gst_buffer_map(outbuf, &out_map, GST_MAP_WRITE)) {
        g_warning("Failed to map output buffer.");
        cudaFree(d_output);
        gst_buffer_unmap(inbuf, &in_map);
        return GST_FLOW_ERROR;
    }

    // Copy back decompressed data
    if (cudaMemcpy(out_map.data, d_output, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        g_warning("CUDA memcpy to host failed.");
        gst_buffer_unmap(outbuf, &out_map);
        cudaFree(d_output);
        gst_buffer_unmap(inbuf, &in_map);
        return GST_FLOW_ERROR;
    }

    // Unmap buffers
    gst_buffer_unmap(inbuf, &in_map);
    gst_buffer_unmap(outbuf, &out_map);

    // Free GPU memory
    cudaFree(d_output);

    return GST_FLOW_OK;
}

// Register plugin
GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    fdctdecompress,
    "Fast DCT Decompression Plugin",
    plugin_init,
    "1.0",
    "LGPL",
    "GStreamer",
    "https://gstreamer.freedesktop.org/"
)
