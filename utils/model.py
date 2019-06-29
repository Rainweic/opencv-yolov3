import cv2 as cv

def load_model(args):
    '''
    load model
    '''
    # Load names of classes
    print("loading coco_names...")
    classesFile_path = args.coco_names
    classes = load_classesFile(classesFile_path)

    ## Give the configuration and weight files for the model and load the network using them.
    print("loading yolov3 config file & weight...")
    modelConfiguration = args.yolov3_cfg;
    modelWeights = args.weight;
    
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    # TODO 尝试设置为GPU
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    print("load suceed!")
    return net, classes

def load_classesFile(path):
    '''
    load classes file
    '''
    classes = None
    with open(path, "rt") as f:
        classes = f.read().rstrip('\n').split('\n')
        classes = [ item for item in classes if item[0] != "#"]
    return classes

