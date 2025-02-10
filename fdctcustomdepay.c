#include "fdctcustomdepay.h"
#include <gst/gst.h>

#define PACKAGE "fastdct"

static gboolean plugin_init(GstPlugin *plugin);

GST_DEBUG_CATEGORY_STATIC(gst_fdct_custom_depay_debug);
#define GST_CAT_DEFAULT gst_fdct_custom_depay_debug

struct _GstFdctCustomDepay {
    GstElement parent;
};

G_DEFINE_TYPE(GstFdctCustomDepay, gst_fdct_custom_depay, GST_TYPE_ELEMENT)

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

// Class initialization
static void gst_fdct_custom_depay_class_init(GstFdctCustomDepayClass *klass) {
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);

    // Add metadata
    gst_element_class_set_metadata(element_class,
        "Fast DCT RTP Depayloader",  // Long name (must be non-empty)
        "Codec/Depayloader",        // Classification
        "Extracts Fast DCT data from RTP packets",  // Description
        "Your Name <your.email@example.com>");  // Author

    GST_DEBUG_CATEGORY_INIT(gst_fdct_custom_depay_debug, "fdctcustomdepay", 0, "RTP FFT Depayloader");
}

// Instance initialization
static void gst_fdct_custom_depay_init(GstFdctCustomDepay *self) {
    GstPad *srcpad, *sinkpad;

    // Corrected function calls: use pad names ("sink" and "src")
    sinkpad = gst_pad_new_from_static_template(&sink_pad_template, "sink");
    gst_element_add_pad(GST_ELEMENT(self), sinkpad);

    srcpad = gst_pad_new_from_static_template(&src_pad_template, "src");
    gst_element_add_pad(GST_ELEMENT(self), srcpad);
}

// Plugin entry point
static gboolean plugin_init(GstPlugin *plugin) {
    if (!gst_element_register(plugin, "fdctcustomdepay", GST_RANK_NONE, GST_TYPE_FDCT_CUSTOM_DEPAY)) {
        g_warning("Failed to register fdctcustomdepay.");
        return FALSE;
    }
    return TRUE;
}

// Register the plugin using GST_PLUGIN_DEFINE
GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    fdctcustomdepay,
    "Fast DCT RTP Depayloader Plugin",
    plugin_init,
    "1.0",
    "LGPL",
    "GStreamer",
    "https://gstreamer.freedesktop.org/"
)
