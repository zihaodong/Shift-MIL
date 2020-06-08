#Python
import numpy as np
import os
import configparser
from matplotlib import pyplot as plt
#Keras
from keras.models import model_from_json
from keras.models import Model
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys
sys.path.insert(0, './lib/')
# help_functions.py
from help_functions import *
# extract_patches.py
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import paint_border
#from extract_patches import kill_border
from extract_patches import pred_only_FOV
from extract_patches import get_data_testing
from extract_patches import get_data_testing_overlap
# pre_processing.py
from pre_processing import my_PreProc
from PIL import Image


#========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('configuration.txt')

#model name
name_experiment = config.get('experiment name', 'name')
path_experiment = './' +name_experiment +'/'

def pred_img(path_im,path_gt):
	new_pred_imgs = []
	new_pred_masks = []
	files = os.listdir(path_im)

	for File in files:		
		im = np.array(Image.open(path_im+File).convert('L'))
		gt = np.array(Image.open(path_gt+File).convert('L'))

		for x in range(im.shape[0]):
			for y in range(im.shape[1]):
				binary_data = im[x][y] / 255
				gt_data = gt[x][y] / 255
				new_pred_imgs.append(binary_data)
				new_pred_masks.append(gt_data)
	# print(new_pred_imgs)
	new_pred_imgs = np.asarray(new_pred_imgs)
	new_pred_masks = np.asarray(new_pred_masks)

	return new_pred_imgs, new_pred_masks

print("\n\n========  Evaluate the results =======================")
#predictions only inside the FOV
# y_scores, y_true = pred_only_FOV(pred_imgs,gtruth_masks, test_border_masks)  #returns data only inside the FOV
# y_scores, y_true = pred_only_FOV(pred_imgs,gtruth_masks)

y_scores_ShiftMIL, y_true_ShiftMIL = pred_img('test/eval_pred/','test/eval_label/')


#1. Area under the ROC curve
fpr_oursmil, tpr_oursmil, thresholds_oursmil = roc_curve((y_true_ShiftMIL), y_scores_ShiftMIL)
AUC_ROC_OURSMIL = roc_auc_score(y_true_ShiftMIL, y_scores_ShiftMIL)
print(("\nArea under the ROC curve: " +str(AUC_ROC_OURSMIL)))

roc_curve = plt.figure()
plt.grid()
plt.plot(fpr_oursmil,tpr_oursmil,'-',label='pw-Unet+Shift-MIL (AUC = %0.4f)' % AUC_ROC_OURSMIL)

plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig(path_experiment+"ROC.png")



#2. Precision-recall curve
precision_oursmil, recall_oursmil, threshold_oursmil = precision_recall_curve((y_true_ShiftMIL), y_scores_ShiftMIL)
precision_oursmil = np.fliplr([precision_oursmil])[0]
recall_oursmil = np.fliplr([recall_oursmil])[0]

AUC_prec_rec_oursmil = np.trapz(precision_oursmil,recall_oursmil)
print(("\nArea under Precision-Recall curve: " +str(AUC_prec_rec_oursmil)))

prec_rec_curve = plt.figure()
plt.grid()
plt.plot(recall_oursmil,precision_oursmil,'-',label='Ours+mil (AUC = %0.4f)' % AUC_prec_rec_oursmil)

plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.savefig(path_experiment+"Precision_recall.png")

