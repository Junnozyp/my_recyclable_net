import cv2
import os
from model_build import TRASH_DICT, TRASH_KIND
from model_build import build_general_model, build_prune_model
import time
import numpy as np


model = build_general_model()

# cap = cv2.VideoCapture(1)
# # state marker
# detect_label = False
# record_path = 'C:/Users/11515/Desktop/ML/RecycleNet-master/Linux_project/image_record'
# image_tested = len(os.listdir(record_path))
# datetime = str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
# count = 0


def get_trash_kind(num):
    trash_kind = TRASH_DICT.get(num)
    trash_class = TRASH_KIND.get(trash_kind)
    return trash_class + " (" + trash_kind + ")"


while (1):

    assert model is not None
    ret, frame = cap.read()

    if detect_label is False:
        cv2.imshow('capture', frame)

    if cv2.waitKey(1) == ord('d'):
        print('Starting classify the moment image: ')
        detect_label = True

        count += 1
        cv2.imwrite(os.path.join(record_path, datetime + '-' + str(image_tested + count) + '.jpg'), frame)

        start = time.process_time()
        image_to_test = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('capture', image_to_test)
        image_to_test = np.reshape(image_to_test, (1, 224, 224, 3))
        prediction = model.predict(image_to_test)

        # first two predictions
        pred_label = np.argsort(prediction[0])[::-1][0:2]

        end = time.process_time()

        first_pred_pro = sorted(prediction[0], reverse=True)[0]
        second_pred_pro = sorted(prediction[0], reverse=True)[1]
        # pre_processed referrence

        if np.abs(first_pred_pro - second_pred_pro) < 0.50:
            print('First prediction: ', get_trash_kind(str(pred_label[0])),
                  '  with prob :', first_pred_pro)
            print('Second prediction: ', get_trash_kind(str(pred_label[1])),
                  '  with prob :', second_pred_pro)
        else:
            print('Final prediction: ', get_trash_kind(str(pred_label[0])),
                  '  with prob :', first_pred_pro)

        print('Time consuming for inferrence : ', end - start)

        detect_label = False
    if cv2.waitKey(1) == ord('e'):
        print('exit!')
        break

cap.release()
cv2.destroyAllWindows()
