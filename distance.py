
# coding: utf-8

# In[1]:

import numpy as np
import scipy.io as sio
import scipy
import scipy.spatial
from scipy.linalg import norm

mat_contents = sio.loadmat('umist_cropped.mat')
facedat = mat_contents['facedat']

people = facedat[0]
people1 = facedat[0][0]
people2 = facedat[0][1]
people1_face1 = facedat[0][0][:,:,0]
people2_face2 = facedat[0][1][:,:,1]

vec_people1_face1 = people1_face1.flatten()
vec_people2_face2 = people2_face2.flatten()
pixels=112*92


# In[2]:

people1_pic_size = facedat[0][0][:][91][1].size
print people1_pic_size
mat_people1 = np.zeros((people1_pic_size,pixels))
for i in range(0,people1_pic_size):
        mat_people1[i]= facedat[0][0][:,:,i].flatten()# mat_people1 是描述 class1 的大矩阵
vec_people1 = mat_people1.flatten()


people2_pic_size = facedat[0][1][:][91][1].size
print people2_pic_size
mat_people2 = np.zeros((people2_pic_size,pixels))
for i in range(0,people2_pic_size):
        mat_people2[i]= facedat[0][1][:,:,i].flatten()# mat_people1 是描述 class2 的大矩阵
vec_people2 = mat_people2.flatten()

def cal_distance_mat_people12(mat_peoplei,mat_peoplej):
    sizei = mat_peoplei.shape[0]
    sizej = mat_peoplej.shape[0]
    size = sizei+sizej
    print size
    mat_dist = np.zeros((size,size),dtype='f') 
    mat_people12=np.row_stack((mat_peoplei,mat_peoplej))
    for i in range(0,size):
        for j in range(0,size):
            mat_dist[i][j]= scipy.spatial.distance.euclidean(mat_people12[i],mat_people12[j])
#             elif (i < ) and (j >= sizei):
#                 mat_dist[i][j]= scipy.spatial.distance.euclidean(mat_peoplei[i],mat_peoplej[j-sizei])
#             elif (i >= sizei) and (j < sizei):
#                 mat_dist[i][j]= scipy.spatial.distance.euclidean(mat_peoplei[i-sizei],mat_peoplej[j])
#             elif (i >= sizei) and (j >= sizej):
#                 mat_dist[i][j]= scipy.spatial.distance.euclidean(mat_peoplei[i-sizei],mat_peoplej[j-sizei])
    return mat_dist # the output is the distance matrix of the people1'face1 and people2'face2 




# In[3]:

#计算class1的类内距离矩阵 ？ 封成函数？ 

def cal_mat_class_distance(mat_peoplei,distance_method):  #（描述第i个class的矩阵 size*10340） 计算类内样本距离，类内样本距离向量
    rows = mat_peoplei.shape[0]
    dist = 0
    num = 0
    vec_class_distance = np.zeros((rows*(rows-1)/2),dtype='f')
    for i in range(0,rows-1):
        for j in range(i+1,rows):
            result = distance_method(mat_peoplei[i],mat_peoplei[j])
            vec_class_distance[num] = result
            dist += result
            num += 1          
    return dist,vec_class_distance


# In[4]:

def sci_norm(mat_person_1,mat_person_2):
    return scipy.spatial.distance.norm((mat_person_1,mat_person_2))
def euclidean(mat_person_1,mat_person_2):
    return scipy.spatial.distance.euclidean(mat_person_1,mat_person_2)
def cosine(mat_person_1,mat_person_2):
    return scipy.spatial.distance.cosine(mat_person_1,mat_person_2)
def pdist(mat_person_1,mat_person_2):
    return scipy.spatial.distance.pdist(np.row_stack((mat_person_1,mat_person_2)))
def lp1(mat_person_1,mat_person_2):
    return norm(mat_person_1-mat_person_2,1)
def angular(array1,array2):
    return norm(array1/norm(array1)-array2/norm(array2))


# In[5]:

def cal_avg_mat_class_distance (mat_people,distance_method): #计算类内样本平均距离
    dist = 0
    num = 0
    for item in mat_people:
        people_item_pic_size = item[:][91][1].size
        num += people_item_pic_size * (people_item_pic_size-1)/2
        # mat_people_item是描述 第item个class 的大矩阵
        mat_people_item = mat2vecs(mat_people_item)
        result_dist,result_vetor =  cal_mat_class_distance(mat_people_item,distance_method)
        dist += result_dist
    av_dist = dist / num
    return av_dist



# In[86]:

mat_dist=cal_distance_mat_people12(mat_people1,mat_people2)

#计算class1 （people1）的类内距离标准差
dist1,vec_class_distance_people1 = cal_mat_class_distance(mat_people1,euclidean) 
std1 = np.std(vec_class_distance_people1)
#计算class2 （people2）的类内距离标准差
dist2,vec_class_distance_people2 = cal_mat_class_distance(mat_people2,euclidean) 
std2 = np.std(vec_class_distance_people2)
std12 = (std1+std2)/2
print std12


# In[87]:

mat_W = np.exp(-mat_dist/std12) #mat_W是 similarity matrix  核函数为 exp(-d/std)
print mat_W


# In[97]:

import nmcut as nc
import networkx as nx
netx = nx.Graph(mat_W)
# eigen_value,vector=nc.ncut(mat_W,72)
# print eigen_value
# print vector
v_partition = nc.normalized_min_cut(netx)
# colors = np.zeros((len(v_partition), 3)) + 1.0
# colors[:, 2] = np.where(v_partition >= 0, 1.0, 0)
# nx.draw(netx, node_color=colors)
print v_partition


# In[90]:

result = v_partition
truth = np.column_stack((np.array([[1]*38]),np.array([[-1]*35])))
x= np.multiply(result,truth)
fenmu = (x + 1)/2

fenmu = fenmu.sum()
accurancy = fenmu / 73.0
print accurancy
# result = vector.diagonal()
# print result
# result = np.sign(result)
# print result


# In[10]:

def cal_distance_between_2_class(mat_person_1,mat_person_2,distance_method):
    distance = 0.0
    for i in range(len(mat_person_1)):
        for j in range(len(mat_person_2)):
            distance += distance_method(mat_person_1[i],mat_person_2[j])    
    return distance


# In[11]:

def cal_avg_distance_between_class(mat_people,distance_method):
    distance = 0.0
    num = len(mat_people)
    num_of_distance = 0
    for i in range(0,num-1):
        for j in range(i+1,num):
            mat_person_i = mat2vecs(mat_people[i])
            mat_person_j = mat2vecs(mat_people[j])
            distance += cal_distance_between_2_class(mat_person_i,mat_person_j,distance_method)
            num_of_distance += len(mat_person_i) * len(mat_person_j)
    return distance / num_of_distance


# In[12]:

def mat2vecs(mat_person):
    people_pic_size = mat_person[:][91][1].size
    # print people_pic_size
    mat_people = np.zeros((people_pic_size,pixels))
    for i in range(0,people_pic_size):
        mat_people[i]= mat_person[:,:,i].flatten()# mat_people1 是描述 class的大矩阵
    return mat_people


# In[13]:

def eval_avg_distance(mat_people,distance_method):
    avg_mat_class_distance = cal_avg_mat_class_distance(mat_people,distance_method) # 平均类内样本距离
    avg_distance_between_class = cal_avg_distance_between_class(mat_people,distance_method) # 平均类间样本距离
    return avg_mat_class_distance / avg_distance_between_class


# In[14]:

print cal_avg_distance_between_class(facedat[0],euclidean)


# In[15]:

print cal_avg_distance_between_class(facedat[0],lp1)


# In[22]:

print cal_avg_distance_between_class(facedat[0],cosine)


# In[23]:

print cal_avg_distance_between_class(facedat[0],angular)


# In[ ]:



