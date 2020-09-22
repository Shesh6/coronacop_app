from fastai.vision.all import *
import cv2
import argparse

# redefine predict function to run on a single worker
def predict(self, item, rm_type_tfms=None, with_input=False):
        dl = self.dls.test_dl([item], rm_type_tfms=rm_type_tfms, num_workers=0)
        inp,preds,_,dec_preds = self.get_preds(dl=dl, with_input=True, with_decoded=True,n_workers=0)
        i = getattr(self.dls, 'n_inp', -1)
        inp = (inp,) if i==1 else tuplify(inp)
        dec = self.dls.decode_batch(inp + tuplify(dec_preds))[0]
        dec_inp,dec_targ = map(detuplify, [dec[:i],dec[i:]])
        res = dec_targ,dec_preds[0],preds[0]
        if with_input: res = (dec_inp,) + res
        return res

# define default arguments
def_args = {"face"      : "",
            "mask"      : "res34_beard.pkl",
            "stamp"     : "mask_crop.png",
            "write"     : "",
            "thresholds": (0.9,0.5)}

def interp_args():

    # interpret arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--picture", required=True, type=str,
                    help="picture path")
    ap.add_argument("-f", "--face", type=str,
                    default=def_args["face"],
                    help="path to face detection model folder")
    ap.add_argument("-m", "--mask", type=str,
                    default=def_args["mask"],
                    help="path to mask detection model")
    ap.add_argument("-s", "--stamp", type=str,
                    default=def_args["stamp"],
                    help="path to mask stamp image")
    ap.add_argument("-w", "--write", type=str,
                    default=def_args["write"],
                    help="path to write output image")
    ap.add_argument("-t", "--thresholds", nargs='+', type=float,
                    default=def_args["thresholds"],
                    help="probability thresholds (face, mask)")
    args = vars(ap.parse_args())

    return (args)

def run(args):

    # load face detector
    face_detect = cv2.dnn.readNetFromCaffe(args["face"]+'deploy.prototxt.txt',
                                           args["face"]+'res10_300x300_ssd_iter_140000.caffemodel')

    # load mask detector and set device to cpu
    mask_detect = load_learner(args["mask"])
    defaults.device = torch.device('cpu')

    # override predict function to circumvent parallel processing bug in fastai for Windows
    setattr(mask_detect, "predict", predict)

    # run prediction
    masks, photoshop = coronacop(args["picture"], args["stamp"], face_detect, mask_detect, tuple(args["thresholds"]), args["write"])

    # save output picture if requested
    if args["write"] != "null":
        cv2.imwrite(args["write"]+".jpg", photoshop)

    return masks

def coronacop(pic_path, mask_path, face_detect, mask_detect, thresh=(0.9,0.5), write="null"):

    # load image, resize, normalize for face detection
    image = cv2.imread(pic_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                            (300, 300), (104.0, 177.0, 123.0)) # values from OpenCV

    # forward pass through face detector to get the predicted bounding boxes and confidence values
    face_detect.setInput(blob)
    detections = face_detect.forward()

    # switch to RGB format for further processing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # initiate arrays
    box_array  = [] # pixel-space coordinates (4)
    face_array = [] # pixel values (244,244)
    mask_array = [] # classification confidence (2)

    # for each face detected
    for i in range(0, detections.shape[2]):
        
        # make sure face confidence is above threshold
        if detections[0, 0, i, 2] > thresh[0]:

            # translate bounding boxes to pixel-space and append to box array
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # confound boxes to picture limits
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract face 
            face = image[startY:endY, startX:endX]

            # ensure valid box
            if 0 not in face.shape:

                # resize for mask detector
                face = cv2.resize(face, (224, 224))
                
                # forward pass through mask detector, extract confidences and append to mask array
                masked, unmasked = mask_detect.predict(mask_detect,face)[2]
                mask_array.append(unmasked)

                # make sure mask confidence is above threshold
                if unmasked >= thresh[1]:

                    # append coordinates and face values to arrays
                    box_array.append((startX, startY, endX, endY))
                    face_array.append(face)

    # photoshop masks if requested
    photoshop = None if write == "null" else photoshop_pic(image, box_array, face_array, mask_path) 

    # return mask_array and photoshop
    return (mask_array, photoshop)


def photoshop_pic(image, box_array, face_array, mask_path):

    # copy image
    photoshop = image.copy()
    photoshop = cv2.cvtColor(photoshop, cv2.COLOR_BGR2RGB)

    # load mask stamp and set format to RGB
    mask_png = cv2.imread(mask_path)
    mask_png = cv2.cvtColor(mask_png, cv2.COLOR_BGR2RGB)

    # loop over arrays
    for i in range(len(box_array)):

        # extract coordinates
        (startX, startY, endX, endY) = box_array[i]

        # resize mask stamp to fit
        mask_h = endY-startY
        mask_w = endX-startX
        mask_resize = cv2.resize(mask_png,(mask_w,mask_h))

        # mask over region of interest
        roi = photoshop[startY:endY,startX:endX]
        mask_gray = cv2.cvtColor(mask_resize,cv2.COLOR_RGB2GRAY)
        ret, mask_mask = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask_mask)

        # paste mask onto region of interest
        background = cv2.bitwise_and(roi,roi,mask = mask_inv)
        foreground = cv2.bitwise_and(mask_resize,mask_resize,mask = mask_mask)
        dst = cv2.add(background,foreground)
        photoshop[startY:endY,startX:endX] = dst

    # return final image
    return photoshop

if __name__ == "__main__":
    run(interp_args())