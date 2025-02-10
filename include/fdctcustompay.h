#ifndef __GST_FDCT_CUSTOM_PAY_H__
#define __GST_FDCT_CUSTOM_PAY_H__

#include <gst/gst.h>

G_BEGIN_DECLS

#define GST_TYPE_FDCT_CUSTOM_PAY (gst_fdct_custom_pay_get_type())
#define GST_FDCT_CUSTOM_PAY(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_FDCT_CUSTOM_PAY, GstFdctCustomPay))
#define GST_FDCT_CUSTOM_PAY_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_FDCT_CUSTOM_PAY, GstFdctCustomPayClass))
#define GST_IS_FDCT_CUSTOM_PAY(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_FDCT_CUSTOM_PAY))
#define GST_IS_FDCT_CUSTOM_PAY_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_FDCT_CUSTOM_PAY))

typedef struct _GstFdctCustomPay GstFdctCustomPay;
typedef struct _GstFdctCustomPayClass GstFdctCustomPayClass;

struct _GstFdctCustomPay {
    GstElement parent;
};

struct _GstFdctCustomPayClass {
    GstElementClass parent_class;
};

GType gst_fdct_custom_pay_get_type(void);

G_END_DECLS

#endif /* __GST_FDCT_CUSTOM_PAY_H__ */
