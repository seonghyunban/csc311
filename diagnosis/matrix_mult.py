import random
import numpy


def generate_random_matrix(n: int, m: int):
    """
    Genereate a random matrix
    :param n: row number
    :param m: column number
    :return: random matrix
    """
    return [[random.randint(-100, 100) for i in range(m)] for j in range(n)]

def generate_empty_matrix(n: int, m: int):
    """
    Generate an empty matrix
    :param n: row number
    :param m: column number
    :return: empty matrix
    """
    return [[0 for i in range(m)] for j in range(n)]

class Matrix:

    def __init__(self, content: list[list]):
        """
        Constructor for Matrix
        :param content: matrix represented as a nested list.
        """
        self.content = content

    def rep(self):
        """
        :return: matrix in the form of nested list representation
        """
        return self.content

    def mult_with_loop(self, other):
        """
        Multiplication with loops
        :param other: matrix that is going to be multiplied to this matrix
        :return: result of the multiplication operation
        """
        self_mat = self.rep()
        other_mat = other.rep()

        n = len(self_mat)
        m = len(other_mat)
        k = len(other_mat[0])

        result_mat = generate_empty_matrix(n, k)

        for x in range(k):
            for i in range(n):
                result_mat[i][x] = sum([self_mat[i][j] * other_mat[j][x] for j in range(m)])

        return Matrix(result_mat)

    def mult_with_numpy(self, other):
        """
        Multiplication with numpy.
        :param other: the matrix to be multiplied to this matrix
        :return: result of the multiplication operation.
        """
        numpy_result = numpy.dot(numpy.array(self.rep()), numpy.array(other.rep()))
        return Matrix(numpy_result.tolist())

    def equals(self, other):
        """
        Check if this matrix is equal to the other matrix.
        :param other: the other matrix
        :return: boolean
        """
        self_mat = numpy.array(self.rep())
        other_mat = numpy.array(other.rep())

        return numpy.array_equal(self_mat, other_mat)



n = 3
m = 2
k = 4

matrix1 = Matrix(generate_random_matrix(n, m))
matrix2 = Matrix(generate_random_matrix(m, k))

multmat1 = matrix1.mult_with_loop(matrix2)
multmat2 = matrix1.mult_with_numpy(matrix2)

print(multmat1.equals(multmat2))













