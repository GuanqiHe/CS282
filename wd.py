import cvxpy as cp
import numpy as np
import img_lib
import matplotlib.pyplot as plt

dataset_path = "MNIST\\decompress_mnist\\"

def c_mat_img(x,y):
    xi,xj = x.shape
    return np.concatenate([point_to_img_cost_mat(i,j,x,y).reshape(-1,1) for j in range(xj) for i in range(xi)],1)

def point_to_img_cost_mat(xi,xj,x,y):
    yi,yj = y.shape
    return np.array([ [point_cost(xi,xj,x,i,j,y) for j in range(yj)] for i in range(yi)])

def point_cost(xi,xj,x,yi,yj,y):
    return abs(xi-yi)+abs(xj-yj)             

def WD_prob(s,d:np.array, c:np.array):
    if type(s) is cp.Variable: 
        s1d = cp.vec(s)
    else:
        s1d = s.reshape((-1,1)).squeeze()
    d1d = d.reshape((-1,1)).squeeze()
    print(s1d.size)
    x = cp.Variable((s1d.size,d1d.size))
    con_s = [cp.sum(x, 0) == s1d]
    con_d = [cp.sum(x, 1) == d1d]
    con_x = [x>=0]
    obj = cp.sum(cp.multiply(c,x))

    return {"obj":obj,"con":con_s+con_d+con_x,"var":{"x":x}}

def WDE_prob(s,d:np.array, c:np.array, eliminate_punish = 10.0):
    '''
    covert constrain sum(x,1) = vec(d) to punishment 
    '''
    if type(s) is cp.Variable: 
        s1d = cp.vec(s)
    else:
        s1d = s.reshape((-1,1)).squeeze()
    d1d = d.reshape((-1,1)).squeeze()
    print(s1d.size)
    x = cp.Variable((s1d.size,d1d.size))
    con_s = [cp.sum(x, 0) == s1d]
    con_x = [x>=0]
    obj = cp.sum(cp.multiply(c,x))+cp.sum(cp.abs(cp.sum(x, 1)-d1d))*eliminate_punish

    return {"obj":obj,"con":con_s+con_x,"var":{"x":x}}



def WD_solve(s:np.array,d:np.array, c:np.array):

    prob_dict = WD_prob(s,d,c)
    prob = cp.Problem(cp.Minimize(prob_dict["obj"]),prob_dict["con"])
    prob.solve()

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(prob_dict["var"]["x"].value)
    print("A dual solution is")
    print(prob.constraints[0].dual_value)

    plt.imshow(prob_dict["var"]["x"].value)
    plt.colorbar()
    plt.show()



def WBP_solve(d_list):
    s = cp.Variable(d_list[0].shape,"s")
    con = []
    obj = 0
    c = c_mat_img(s,d_list[0])
    for d in d_list:
        t = cp.Variable()
        wd_dict = WDE_prob(s, d, c)
        obj+=t
        con+=wd_dict["con"]+[t>=wd_dict["obj"]]
    prob = cp.Problem(cp.Minimize(obj),con)
    prob.solve("ECOS")

    print(obj)
    print(con)
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(s.value)
    print("A dual solution is")
    print(prob.constraints[0].dual_value)

    plt.imshow(s.value)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    # img0 = img_lib.getimg_downsample("0.png",4)
    # img1 = img_lib.getimg_downsample("3.png",4)
    # print(np.average(img0),np.average(img1))
    print(cp.installed_solvers())
    img_list = img_lib.getimg_dataset(dataset_path+"train\\0", 1, 2)
    img_list = [i[1] for i in img_list]


    # c = c_mat_img(img0,img1)

    WBP_solve(img_list)

    # WD_solve(img0,img1,c)
    # print("x",img0)
    # print("y",img1)

    # print(c_mat_img(img0, img1))