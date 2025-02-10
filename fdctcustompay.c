#include "fdctcustompay.h"
#include <gst/gst.h>

// Define PACKAGE macro for the build system
#define PACKAGE "fastdct"

// Ensure you have this macro to define the element's type
G_DEFINE_TYPE(GstFdctCustomPay, gst_fdct_custom_pay, GST_TYPE_ELEMENT);

static gboolean plugin_init(GstPlugin *plugin);

GST_DEBUG_CATEGORY_STATIC(gst_fdct_custom_pay_debug);
#define GST_CAT_DEFAULT gst_fdct_custom_pay_debug

static GstStaticPadTemplate src_pad_template =
    GST_STATIC_PAD_TEMPLATE("src",
                            GST_PAD_SRC,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS("application/x-rtp, media=video"));

static GstStaticPadTemplate sink_pad_template =
    GST_STATIC_PAD_TEMPLATE("sink",
                            GST_PAD_SINK,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS("video/x-raw, format=BGR"));

static void gst_fdct_custom_pay_class_init(GstFdctCustomPayClass *klass) {
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);

    // Add metadata
    gst_element_class_set_metadata(element_class,
        "Fast DCT RTP Payloader",   // Long name (must be non-empty)
        "Codec/Payloader",         // Classification
        "Encapsulates Fast DCT data into RTP packets",  // Description
        "Your Name <your.email@example.com>");  // Author

    GST_DEBUG_CATEGORY_INIT(gst_fdct_custom_pay_debug, "fdctcustompay", 0, "RTP FFT Payloader");
}

static void gst_fdct_custom_pay_init(GstFdctCustomPay *self) {
    gst_element_add_pad(GST_ELEMENT(self), gst_pad_new_from_static_template(&sink_pad_template, "sink"));
    gst_element_add_pad(GST_ELEMENT(self), gst_pad_new_from_static_template(&src_pad_template, "src"));
}

// Register the plugin using GST_PLUGIN_DEFINE
GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    fdctcustompay,
    "Fast DCT RTP Payloader Plugin",
    plugin_init,
    "1.0",
    "LGPL",
    "GStreamer",
    "https://gstreamer.freedesktop.org/"
)

// Define plugin_init function
static gboolean plugin_init(GstPlugin *plugin) {
    // Register the payloader
    if (!gst_element_register(plugin, "fdctcustompay", GST_RANK_NONE, GST_TYPE_FDCT_CUSTOM_PAY)) {
        g_warning("Failed to register fdctcustompay.");
        return FALSE;
    }
    return TRUE;
}
