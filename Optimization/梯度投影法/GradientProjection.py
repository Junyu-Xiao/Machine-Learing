import numpy as np
import tensorflow as tf
import itertools as its


class GPM:
    """
    梯度投影法：在可行域内求解最优解
    """

    def __init__(self, f_x, A, A_b, E=None, E_b=None):
        """
        :param f_x: TensorFlow运算函数
        :param A: 不等式约束矩阵
        :param E: 等式约束矩阵
        :param A_b: 不等式约束值
        :param E_b: 等式约束值
        """
        self.f_x = f_x
        self.A = A
        self.E = E
        self.A_b = A_b
        self.E_b = E_b

    @staticmethod
    def projection_matrix(matrix):
        """
        :param matrix: 空间矩阵(m*n) --> 矩阵需为行满秩矩阵
        :return: 输出为matrix的零空间的投影矩阵
        """
        matrix_T = matrix.T
        I = np.eye(matrix_T.shape[0])
        Inv = np.linalg.pinv(np.dot(matrix, matrix_T))
        shadow = I - np.dot(np.dot(matrix_T, Inv), matrix)
        return shadow

    @staticmethod
    def gradient_cul(f_x, x):
        """
        :param f_x: 目标函数
        :param x: 变量位置
        :return: 目标函数当前梯度
        """
        with tf.GradientTape(persistent=True) as tape:
            tf_x = tf.Variable(x)
            y = f_x(tf_x)
        gradient = tape.gradient(y, tf_x)
        del tape
        return gradient.numpy()

    @staticmethod
    def best_search(f_x, x, d, min_, max_, lag=100):
        """
        :param f_x: 目标函数
        :param x: 当前x
        :param d: 可行方向
        :param min_: 最小lambda
        :param max_: 最大lambda
        :param lag: 搜索次数
        :return: 最优x
        """
        lag_len = (max_ - min_) / lag
        best_x = x + min_ * d
        min_f = f_x(best_x)
        for i in range(lag):
            search_x = x + (min_ + lag_len * (i + 1)) * d
            search_f = f_x(search_x)
            if search_f <= min_f:
                min_f = search_f
                best_x = search_x
        return best_x

    def point_init(self, e=1e-10):
        """
        初始化可行点
        :return: 初始可行点
        """
        global b_hat
        A = self.A
        E = self.E
        A_b = self.A_b
        E_b = self.E_b

        # 根据等式条件生成初始可行点
        if E is None:
            x = np.zeros([A.shape[1], 1])
        elif E.shape[0] > E.shape[1]:
            print('等式条件错误：等式条件过多')
            return None
        elif E.shape[0] == E.shape[1]:
            if np.abs(np.linalg.det(E)) >= e:
                inv = np.linalg.inv(E)
                x = np.dot(inv, E_b)
                cond = (np.dot(A, x) - A_b).min()
                if cond >= 0:
                    return x
                else:
                    print('等式条件错误：等式结果无法满足不等式')
                    return None
            else:
                inv = np.linalg.pinv(E)
                x = np.dot(inv, E_b)
        else:
            inv = np.linalg.pinv(E)
            x = np.dot(inv, E_b)

        # 逐条更新可行点
        for i in range(A.shape[0]):
            current_cond = A[i: i + 1, :]
            current_b = A_b[i: i + 1, :]
            if i == 0:
                past_group = None
                past_group_b = None
            else:
                past_group = A[:i, :]
                past_group_b = A_b[:i, :]
            cond = (np.dot(current_cond, x) - current_b).min()

            # 可行点判断
            if cond >= 0:
                continue

            # 等式投影矩阵
            if E is None:
                P = np.eye(A.shape[1])
            else:
                P = self.projection_matrix(E)

            # 可行方向
            d = np.dot(P, np.transpose(current_cond, [1, 0]))
            if np.dot(np.transpose(d, [1, 0]), d).sum() < e:
                print('条件错误：条件冲突')
                return None

            # 按投影更新可行点
            while cond < 0:
                lamb_min = (current_b - np.dot(current_cond, x)) / np.dot(current_cond, d)
                if past_group is None:
                    d_hat = np.array([[1.]])
                    b_hat = np.array([[1.]])
                else:
                    d_hat = np.dot(past_group, d)
                    b_hat = past_group_b - np.dot(past_group, x)
                index = d_hat[:, 0] < 0
                lamb_list = b_hat[index] / d_hat[index]

                if max(index) == 1:
                    min_index = np.argmin(lamb_list)
                    lamb_max = lamb_list[min_index].min()
                else:
                    lamb_max = lamb_min

                if lamb_min <= lamb_max:
                    lamb_mean = (lamb_max + lamb_min) / 2
                    x = x + lamb_mean * d
                    cond = (np.dot(current_cond, x) - current_b).min()
                else:
                    x = x + lamb_max * d
                    cond = (np.dot(current_cond, x) - current_b).min()
                    new_equal = past_group[index][[min_index], :]
                    if E is None:
                        M = new_equal
                    else:
                        M = np.vstack(E, new_equal)
                        if M.shape[0] >= M.shape[1]:
                            print('非行满秩矩阵')
                            return None
                    P = self.projection_matrix(M)
                    d = np.dot(P, np.transpose(current_cond, [1, 0]))
                    if np.dot(np.transpose(d, [1, 0]), d).sum() < e:
                        print('条件错误：条件冲突')
                        return None

        return x

    def gradient_projection(self, inside=True, origin_point=None, e=1e-10, lag=10, lr=1, epoch=100, **kwargs):
        """
        :param epoch: 最大迭代次数
        :param lr: 学习率(当梯度方向无限制条件限制时)
        :param lag:　一维搜索的搜索步长
        :param inside:　是否使用内置限制条件
        :param origin_point: 初始可行点
        :param e: 误差允许值
        :return: 可行最优解
        """
        if inside:
            A = self.A
            A_b = self.A_b
            E = self.E
            f_x = self.f_x
        else:
            A = kwargs.get('A')
            A_b = kwargs.get('A_b')
            E = kwargs.get('E')
            f_x = kwargs.get('f_x')

        # 若不提供初始点则使用内置方法生成初始点
        if origin_point is None:
            origin_point = self.point_init(e=e)

        x = origin_point

        status = 0
        step = 0
        while (status == 0) & (step <= epoch):

            # 判断当前可行点是否存在等式条件或处于边界上
            index_11, index_12 = (np.abs(np.dot(A, x) - A_b) <= e)[:, 0], (np.abs(np.dot(A, x) - A_b) > e)[:, 0]
            A_11, A_12 = A[index_11], A[index_12]
            A_b_11, A_b_12 = A_b[index_11], A_b[index_12]

            if E is None:
                if A_11.shape[0] != 0:
                    M = A_11
                    M_size = M.shape[0]
                else:
                    M_size = 0
            else:
                if A_11.shape[0] != 0:
                    M = np.vstack([A_11, E])
                    M_size = M.shape[0]
                else:
                    M = E
                    M_size = M.shape[0]

            # 计算等式投影矩阵
            if M_size == 0:
                P = np.eye(A.shape[1])
            else:
                P = self.projection_matrix(M)

            # 计算梯度方向
            grad = self.gradient_cul(f_x, x)
            # 可行方向
            d = -1 * np.dot(P, grad)

            # 可行方向模长
            d_l2 = np.dot(np.transpose(d, [1, 0]), d).sum()

            # 可行方向判断
            while (d_l2 <= e) & (status == 0):
                if (d_l2 <= e) & (M_size == 0):
                    status = 1
                elif (d_l2 <= e) & (M_size != 0) & (A_11.shape[0] == 0):
                    status = 1
                elif (d_l2 <= e) & (M_size != 0) & (A_11.shape[0] != 0):
                    if E is not None:
                        w = np.dot(np.dot(np.linalg.inv(np.dot(M, np.transpose(M, [1, 0]))), M), grad)
                        u, v = w[:A_11.shape[0]], w[A_11.shape[0]:]
                        kkt_index = (u >= 0)[:, 0]
                        if kkt_index.min() == 1:
                            status = 1
                        else:
                            A_11 = A_11[kkt_index]
                            M = np.vstack([A_11, E])
                            M_size = M.shape[0]
                            P = self.projection_matrix(M)
                            d = -1 * np.dot(P, grad)
                            d_l2 = np.dot(np.transpose(d, [1, 0]), d).sum()
                    else:
                        w = np.dot(np.dot(np.linalg.pinv(np.dot(M, np.transpose(M, [1, 0]))), M), grad)
                        kkt_index = (w >= 0)[:, 0]
                        if kkt_index.min() == 1:
                            status = 1
                        elif kkt_index.max() == 0:
                            M_size = 0
                            P = np.eye(A.shape[1])
                            d = -1 * np.dot(P, grad)
                            d_l2 = np.dot(np.transpose(d, [1, 0]), d).sum()
                        else:
                            A_11 = A_11[kkt_index]
                            M = A_11[:]
                            M_size = M.shape[0]
                            P = self.projection_matrix(M)
                            d = -1 * np.dot(P, grad)
                            d_l2 = np.dot(np.transpose(d, [1, 0]), d).sum()

            if d_l2 > e:
                d_hat = np.dot(A_12, d)
                b_hat = A_b_12 - np.dot(A_12, x)
                lamb_index = (d_hat < 0)[:, 0]
                if lamb_index.max() == 0:
                    lamb_min = 0
                    lamb_max = lr
                else:
                    d_hat = d_hat[lamb_index]
                    b_hat = b_hat[lamb_index]
                    lamb_list = b_hat / d_hat
                    lamb_min = 0
                    lamb_max = lamb_list.min()
                x = self.best_search(f_x, x, d, lamb_min, lamb_max, lag=lag)
            step += 1

        return x

    def out_of_int(self, x):
        A = self.A
        A_b = self.A_b
        E = self.E

        if E is not None:
            print('不支持含有等式条件的转化')
            return None

        x_int = np.floor(x)

        if ((np.dot(A, x_int) - A_b) >= 0).min() == 1:
            return x_int

        for i in range(x_int.shape[0]):
            for item in its.combinations_with_replacement(np.array(range(x_int.shape[0])), i + 1):
                add = np.zeros(x_int.shape)
                for index in item:
                    add[index] += 1
                x_try = x_int + add
                if ((np.dot(A, x_try) - A_b) >= 0).min() == 1:
                    return x_try

        return np.ceil(x)


# 使用实例
if __name__ == '__main__':

    """
    A: 不等式矩阵
    A_b: 不等式偏置项
    E: 等式矩阵
    E_b: 等式偏置项
    f_x: 目标优化函数
    不等式约束条件: Ax >= A_b
    等式约束条件: Ex >= E_b
    输出: 在等式与不等式约束条件下，使得f_x最小的x值
    """
    A = np.array([[0., 0., 0., 0., 4., 3.],
                  [0., 2., 0., 4., 0., 0.],
                  [0., 0., 3., 0., 0., 0.],
                  [1., 1., 0., 0., 0., 1.],
                  [0., 0., 0., 0., 7., 4.],
                  [0., 3., 0., 6., 0., 0.],
                  [0., 0., 5., 0., 0., 0.],
                  [3., 1., 0., 0., 0., 1.]]).astype('float')

    A_b = np.array([[5.],
                    [2.],
                    [0.],
                    [2.],
                    [15.],
                    [6.],
                    [0.],
                    [0.]]).astype('float')
    f_x = lambda x: tf.matmul(tf.transpose(x, [1, 0]), x)
    test = GPM(f_x, A, A_b)
    best_value = test.gradient_projection(e=1e-3, lag=1000, epoch=100)
    int_out = test.out_of_int(best_value)
    print(best_value)
    print(int_out)