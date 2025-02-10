#ifndef DECOMPRESSIONPART_H
#define DECOMPRESSIONPART_H

#include <gst/gst.h>

G_BEGIN_DECLS

#define GST_TYPE_FDCT_DECOMPRESS (gst_fdct_decompress_get_type())

G_DECLARE_FINAL_TYPE(GstFdctDecompress, gst_fdct_decompress, GST, FDCT_DECOMPRESS, GstElement);

G_END_DECLS

#endif /* DECOMPRESSIONPART_H */
