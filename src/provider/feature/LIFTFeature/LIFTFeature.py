from src.provider.feature.LIFTFeature.lift_utils.kp_tools import XYZS2kpList, get_XYZS_from_res_list, kp_list_2_opencv_kp_list, IDX_ANGLE, update_affine
from src.provider.feature.LIFTFeature.lift_utils.filter_tools import apply_learned_filter_2_image_no_theano
from src.provider.feature.LIFTFeature.lift_utils.custom_types import paramGroup, paramStruct, pathConfig
from src.provider.feature.LIFTFeature.lift_utils.dataset_tools.data_obj import data_obj
from src.provider.feature.LIFTFeature.lift_utils.dump_tools import loadh5
from src.provider.feature.LIFTFeature.lift_utils.solvers import model
from copy import deepcopy
import numpy as np
import cv2

class LIFTFeature:

    def __init__(self, 
                 config_file:str='src/utils/configs/models/configs/picc-finetune-nopair.config',
                 model_dir='src/utils/configs/models/picc-best/',
                 model_base_sir='src/utils/configs/models/base',
                 mean_td_dir='src/utils/configs/models/picc-best/mean_std.h5',
                 num_keypoint:int=1000):

        self.config_file = config_file
        self.model_dir = model_dir
        self.mean_std_dir = mean_td_dir
        self.num_keypoint = num_keypoint
        self.model_base_sir = model_base_sir

    def __setup_model(self):
        self.param = paramStruct()
        self.param .loadParam(self.config_file, verbose=False)
        self.pathconf = pathConfig()
        self.pathconf.setupTrain(self.param , 0)
        self.pathconf.result = self.model_dir
    
    def __get_dimention_of_image(self, img_in: np.ndarray):

        return img_in.shape[0], img_in.shape[1]
    
    def __load_mean(self):
        mean_std_dict = loadh5(self.mean_std_dir)
        self.param.online = paramGroup()
        setattr(self.param.online, 'mean_x', mean_std_dict['mean_x'])
        setattr(self.param.online, 'std_x', mean_std_dict['std_x'])

    def __multscale(self, img_in: np.ndarray, image_height, image_width):
        
        # Setup
        self.__setup_model()
        # Multiscale Testing
        scl_intv = getattr(self.param.validation, 'nScaleInterval', 4)
        min_scale_log2 = getattr(self.param.validation, 'min_scale_log2', 1)
        max_scale_log2 = getattr(self.param.validation, 'max_scale_log2', 4)
        # Test starting with double scale if small image
        min_hw = np.min(img_in.shape[:2])
        if min_hw <= 1600:
            # print("INFO: Testing double scale")
            min_scale_log2 -= 1
        # range of scales to check
        num_division = (max_scale_log2 - min_scale_log2) * (scl_intv + 1) + 1
        scales_to_test = 2**np.linspace(min_scale_log2, max_scale_log2,
                                        num_division)

        # convert scale to image resizes
        resize_to_test = ((float(self.param.model.nPatchSizeKp - 1) / 2.0) /
                        (self.param.patch.fRatioScale * scales_to_test))

        # check if resize is valid
        min_hw_after_resize = resize_to_test * np.min(img_in.shape[:2])
        is_resize_valid = min_hw_after_resize > self.param.model.nFilterSize + 1

        # if there are invalid scales and resizes
        if not np.prod(is_resize_valid):
            # find first invalid
            first_invalid = np.where(True - is_resize_valid)[0][0]

            # remove scales from testing
            scales_to_test = scales_to_test[:first_invalid]
            resize_to_test = resize_to_test[:first_invalid]

        # Run for each scale
        test_res_list = []
        for resize in resize_to_test:

            # Just designate only one scale to bypass resizing. Just a single
            # number is fine, no need for a specific number
            param_cur_scale = deepcopy(self.param)
            param_cur_scale.patch.fScaleList = [
                1.0
            ]

            # resize according to how we extracted patches when training
            new_height = int(image_height * resize)
            new_width = int(image_width * resize)
            image = cv2.resize(img_in, (new_width, new_height))

            s_kp_nonlinearity = getattr(self.param.model, 'sKpNonlinearity', 'None')
            test_res = apply_learned_filter_2_image_no_theano(
                image, self.pathconf.result,
                self.param.model.bNormalizeInput,
                s_kp_nonlinearity,
                verbose=True)

            test_res_list += [np.pad(test_res,
                                    int((self.param.model.nFilterSize - 1) / 2),
                                    # mode='edge')]
                                    mode='constant',
                                    constant_values=-np.inf)]

        ##############################################################
        # Non-max suppresion and draw.
        nearby = int(np.round(
            (0.5 * (self.param.model.nPatchSizeKp - 1.0) *
            float(self.param.model.nDescInputSize) /
            float(self.param.patch.nPatchSize))
        ))
        f_nearby_ratio = getattr(self.param.validation, 'fNearbyRatio', 1.0)
        # Multiply by quarter to compensate
        f_nearby_ratio *= 0.25
        nearby = int(np.round(nearby * f_nearby_ratio))
        nearby = max(nearby, 1)

        nms_intv = getattr(self.param.validation, 'nNMSInterval', 2)
        edge_th = getattr(self.param.validation, 'fEdgeThreshold', 10)
        do_interpolation = getattr(self.param.validation, 'bInterpolate', True)

        f_scale_edgeness = getattr(self.param.validation, 'fScaleEdgeness', 0)
        res_list = test_res_list
        XYZS = get_XYZS_from_res_list(res_list, resize_to_test,
                                    scales_to_test, nearby, edge_th,
                                    scl_intv, nms_intv, do_interpolation,
                                    f_scale_edgeness)
        XYZS = XYZS[:self.num_keypoint]

        kp_list = XYZS2kpList(XYZS)

        return kp_list
    
    def __get_key_point(self, img_in: np.array, kp_list):
        # Setup
        self.__setup_model()

        self.param.model.sDetector = 'bypass'
        # This ensures that you don't create unecessary scale space
        self.param.model.fScaleList = np.array([1.0])
        self.param.patch.fMaxScale = np.max(self.param.model.fScaleList)
        # this ensures that you don't over eliminate features at boundaries
        self.param.model.nPatchSize = int(np.round(self.param.model.nDescInputSize) * np.sqrt(2))
        self.param.patch.fRatioScale = (float(self.param.patch.nPatchSize) / float(self.param.model.nDescInputSize)) * 6.0
        self.param.model.sDescriptor = 'bypass'

        # add the mean and std of the learned model to the self.param
        self.__load_mean()

        # -------------------------------------------------------------------------
        # Load data in the test format
        kp_array = np.stack(kp_list)
        test_data_in = data_obj(self.param, img_in, kp_array)

        # -------------------------------------------------------------------------
        # Test using the test function
        _, oris, = model( self.pathconf, self.param, test_data_in, test_mode="ori")

        # update keypoints and save as new
        kp_array = test_data_in.coords
        for idxkp in range(kp_array.shape[0]):
            kp_array[idxkp][IDX_ANGLE] = oris[idxkp] * 180.0 / np.pi % 360.0
            kp_array[idxkp] = update_affine(kp_array[idxkp])
        
        return kp_array
    
    def __get_descriptor(self, img_in: np.array, kp_array, convert_to_uint8):
        # Setup
        self.__setup_model()

        setattr(self.param.model, "descriptor_export_folder", self.model_base_sir)

        # Modify the network so that we bypass the keypoint part and the
        # orientation part
        self.param.model.sDetector = 'bypass'
        # This ensures that you don't create unecessary scale space
        self.param.model.fScaleList = np.array([1.0])
        self.param.patch.fMaxScale = np.max(self.param.model.fScaleList)
        # this ensures that you don't over eliminate features at boundaries
        self.param.model.nPatchSize = int(np.round(self.param.model.nDescInputSize) * np.sqrt(2))
        self.param.patch.fRatioScale = (float(self.param.patch.nPatchSize) / float(self.param.model.nDescInputSize)) * 6.0
        self.param.model.sOrientation = 'bypass'

        # add the mean and std of the learned model to the self.param
        self.__load_mean()

        # -------------------------------------------------------------------------
        # Load data in the test format
        test_data_in = data_obj(self.param, img_in, kp_array)

        # -------------------------------------------------------------------------
        # Test using the test function
        descs, _, = model( self.pathconf, self.param, test_data_in, test_mode="desc")

        if convert_to_uint8:
            descs = descs * 255.0 / 4.0
            descs = np.clip(descs, 0, 255)
            descs = np.array(descs, dtype=np.ubyte)

        kp_list = []
        for i in range(kp_array.shape[0]):
            kp_list.append(kp_array[i])

        return descs, kp_list


    def extract_feature(self, img_in: np.ndarray, convert_to_uint8: bool = False):
        """
        img_in: grayscale image in np array format (H x W)
        return: image_overlay (colored img), keypoint location, keypoints descriptions
        """

        if (len(img_in.shape) != 2):
            img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

        # check size
        image_height, image_width = self.__get_dimention_of_image(img_in)
        
        # Mutiscale test
        kp_list = self.__multscale(img_in, image_height, image_width)

        # Get keypoint
        kp_array = self.__get_key_point(img_in, kp_list)

        # Get descriptor
        descs, kp_list = self.__get_descriptor(img_in, kp_array, convert_to_uint8)
        
        # Convert it
        kp_cv2 = kp_list_2_opencv_kp_list(kp_list)

        return np.array([(kp.pt[0], kp.pt[1]) for kp in kp_cv2]), descs

    def normalize(self, count_inv, pts):

        return np.dot(count_inv, np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1).T).T[:, 0:2]

    def denormalize(self, count, pt):

        ret = np.dot(count, np.array([pt[0], pt[1], 1.0]))
        ret /= ret[2]
        return int(round(ret[0])), int(round(ret[1]))