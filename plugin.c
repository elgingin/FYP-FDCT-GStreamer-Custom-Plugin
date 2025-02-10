#define PACKAGE "fastdct"
#include <gst/gst.h>
#include "compressionpart.h"
#include "decompressionpart.h"
#include "fdctcustompay.h"
#include "fdctcustomdepay.h"

// Declare plugin_init before GST_PLUGIN_DEFINE
static gboolean plugin_init(GstPlugin *plugin);

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    fdctplugin, // Plugin name (this should be the same for gst-inspect and gst-launch)
    "Fast DCT Compression Plugin",
    plugin_init,
    "1.0",
    "LGPL",
    "GStreamer",
    "https://gstreamer.freedesktop.org/"
)

// Define plugin_init function
static gboolean plugin_init(GstPlugin *plugin) {
    // Register each element with the correct element type
    if (!gst_element_register(plugin, "fdctcompress", GST_RANK_NONE, GST_TYPE_FDCT_COMPRESS)) {
        g_warning("Failed to register fdctcompress.");
        return FALSE;
    }
    if (!gst_element_register(plugin, "fdctdecompress", GST_RANK_NONE, GST_TYPE_FDCT_DECOMPRESS)) {
        g_warning("Failed to register fdctdecompress.");
        return FALSE;
    }
    if (!gst_element_register(plugin, "fdctcustompay", GST_RANK_NONE, GST_TYPE_FDCT_CUSTOM_PAY)) {
        g_warning("Failed to register fdctcustompay.");
        return FALSE;
    }
    if (!gst_element_register(plugin, "fdctcustomdepay", GST_RANK_NONE, GST_TYPE_FDCT_CUSTOM_DEPAY)) {
        g_warning("Failed to register fdctcustomdepay.");
        return FALSE;
    }

    return TRUE;
}
