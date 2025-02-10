#ifndef FDCTCUSTOMDEPAY_H
#define FDCTCUSTOMDEPAY_H

#include <gst/gst.h>

G_BEGIN_DECLS

#define GST_TYPE_FDCT_CUSTOM_DEPAY (gst_fdct_custom_depay_get_type())

G_DECLARE_FINAL_TYPE(GstFdctCustomDepay, gst_fdct_custom_depay, GST, FDCT_CUSTOM_DEPAY, GstElement);

G_END_DECLS

#endif /* FDCTCUSTOMDEPAY_H */
