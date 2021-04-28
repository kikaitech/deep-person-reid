from torchreid.utils import FeatureExtractor
import numpy
import os
import cv2

extractor = FeatureExtractor(
    model_name='osnet_x0_25',
    model_path='/home/kikai/Desktop/nobi/deep-person-reid/log/osnet_x0_25_market1501_softmax/model/model.pth.tar-180',
    device='cuda'
)

image_list = []
image_feature_s = {}
# query = "/home/kikai/Desktop/nobi/deep-person-reid/test2.jpg"
# q_feature = extractor(query).cpu().numpy()

for i in os.listdir("/home/kikai/Desktop/nobi/deep-person-reid/data/market1501/Market-1501-v15.09.15/bounding_box_test/"):
    path = "/home/kikai/Desktop/nobi/deep-person-reid/data/market1501/Market-1501-v15.09.15/bounding_box_test/" + i
    image_list.append(path)

for image in image_list:
    feature = extractor(image).cpu().numpy()
    image_feature_s[image] = feature


all_time = 0
true = 0
false = 0

best_dist = 999999999
best_result = ""
best_result_i = ""
best_image = ""
list_dist = {}
for i in os.listdir("/home/kikai/Desktop/nobi/deep-person-reid/data/market1501/Market-1501-v15.09.15/query/"):
    all_time += 1
    print(all_time)
    query = "/home/kikai/Desktop/nobi/deep-person-reid/data/market1501/Market-1501-v15.09.15/query/" + i
    resident_id = i.split("_")[2]
    q_feature = extractor(query).cpu().numpy()
    best_dist = 999999999
    for image in image_feature_s:
        dist = numpy.linalg.norm(q_feature - image_feature_s[image])
        if dist != 0:
            if dist < best_dist:
                best_dist = dist
                best_result = image.split("/")[-1].split("_")[2]
                best_result_i = image
    print(best_dist)

    query_i = cv2.imread(query)
    result_i = cv2.imread(best_result_i)
    cv2.imshow(query.split("/")[-1], query_i)
    cv2.imshow(best_result_i.split("/")[-1], result_i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if resident_id == best_result:
        true += 1
    else:
        false += 1

print(all_time)
print(true)
print(false)
