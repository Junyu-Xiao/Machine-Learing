import numpy as np
import tensorflow as tf


class ECOP:
    """
    梯度投影法：在可行域内求解最优解
    """

    def __init__(self, f_x, A, A_b, E=None, E_b=None):
        """
        :param f_x: tensorflow运算函数
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
        Inv = np.linalg.inv(np.dot(matrix, matrix_T))
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
        return gradient

    @staticmethod
    def one_search(f_x, x, d, min_, max_, lag):
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
        A = self.A
        E = self.E
        A_b = self.A_b
        E_b = self.E_b

        # 根据等式条件生成初始可行点
        if E is None:
            x = np.zeros([A.shape[1], 1])
        elif E.shape[0] > E.shape[1]:
            return '等式条件错误：等式条件过多'
        elif E.shape[0] == E.shape[1]:
            if np.abs(np.linalg.det(E)) >= e:
                inv = np.linalg.inv(E)
                x = np.dot(inv, E_b)
                cond = (np.dot(A, x) - A_b).min()
                if cond >= 0:
                    return x
                else:
                    return '等式条件错误：等式结果无法满足不等式'
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
                past_group = np.ones([1, A.shape[1]])
                past_group_b = np.array([[-99999.]])
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
                return '条件错误：条件冲突'

            # 按投影更新可行点
            while cond < 0:
                lamb_min = (current_b - np.dot(current_cond, x)) / np.dot(current_cond, d)
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
                            return '非行满秩矩阵'
                    P = self.projection_matrix(M)
                    d = np.dot(P, np.transpose(current_cond, [1, 0]))
                    if np.dot(np.transpose(d, [1, 0]), d).sum() < e:
                        return '条件错误：条件冲突'

        return x

    def gradient_projection(self, origin_point=None, e=1e-10):
        """
        :param origin_point: 初始可行点
        :param e: 误差允许值
        :return: 可行最优解
        """
        A = self.A
        A_b = self.A_b
        E = self.E
        E_b = self.E_b
        f_x = self.f_x

        # 若不提供初始点则使用内置方法生成初始点
        if origin_point is None:
            origin_point = self.point_init(e=e)

