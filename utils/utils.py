import cv2 as cv
import numpy as np

def draw_bbox_label(image, classes, classIds, bboxes, confidences):
    for i in range(len(classIds)):
        classId = classIds[i]
        bbox = bboxes[i]
        confidence = confidences[i]

        cv.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255))

        label = '%.2f' % confidence
        # Get the label for the class name and its confidence
        if classes:
            assert(classId < len(classes))
            label = '%s:%s' % (classes[classId], label)
        
        #Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x = bbox[0]
        y = max(bbox[1], labelSize[1])
        cv.putText(image, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

def process_video(args, net, classes):
    cap = cv.VideoCapture(eval(args.video))

    # Get video information
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    save_cap = cv.VideoWriter("./output.mp4", cv.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, (width,height))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print("Video is over")
            print("The video is output.mp4")
            break
        # deal with the frame
        classIds, bboxes, confidences = process_img(args, net, frame, classes)
        # draw bbox and text
        draw_bbox_label(frame, classes, classIds, bboxes, confidences)
        
        # Show frame
        cv.imshow("video", frame)
        cv.waitKey(0)

        # Save video 
        save_cap.write(frame)
    cap.release()
    save_cap.release()
    cv.destroyAllWindows()

def process_img(args, net, img, classes):
    """
    get bbox by process img
    """
    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(img, 1/255, (args.inpWidth, args.inpHeight), [0,0,0], 1, crop=False)
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(__getOutputsNames(net))
    # Remove the bounding boxes with low confidence
    classIds, bboxes, confidences = __postprocess(img, outs, args.confThreshold, args.nmsThreshold)
    if args.image:
        draw_bbox_label(img, classes, classIds, bboxes, confidences)
        cv.imwrite("output.jpg", img)
    return classIds, bboxes, confidences

def __getOutputsNames(net):
    """
    Get the names of the output layers
    """
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def __postprocess(frame, outs, confThreshold, nmsThreshold):
    """
    Remove the bounding boxes with low confidence using non-maxima suppression
    And return bbox list

    bbox like:  [left_top_x, left_top_y, width, height, center_x, center_y]
    """
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    boxes_with_centerxy = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left_top_x = int(center_x - width / 2)
                left_top_y = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left_top_x, left_top_y, width, height])
                boxes_with_centerxy.append([left_top_x, left_top_y, width, height, center_x, center_y])
 
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    #use indices to select bbox and return them
    boxes_with_centerxy = [boxes_with_centerxy[i[0]] for i in indices]
    classIds = [classIds[i[0]] for i in indices]
    confidences = [confidences[i[0]] for i in indices]

    return classIds, boxes_with_centerxy, confidences
