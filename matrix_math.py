import math

class ShapeException(Exception):
    pass


def shape(some_matrix):
    if isinstance(some_matrix[0], list):
        return len(some_matrix),len(some_matrix[0])
    else:
        return (len(some_matrix),)


def shape_vectors(some_vector):
    """shape should take a vector or matrix and return a tuple with the
    number of rows (for a vector) or the number of rows and columns
    (for a matrix.)"""
    return shape(some_vector)


def vector_add(x,y):
    """
    [a b]  + [c d]  = [a+c b+d]

    Matrix + Matrix = Matrix
    """
    if shape(x) != shape(y):
        raise ShapeException
    else:
        return [ x[i]+y[i] for i in range(len(x))]


def vector_add_is_communicative(x,y):
    return vector_add(x,y) == vector_add(y,x)


def vector_add_checks_shapes(x,y):
    vector_add(x,y)


def vector_sub(x,y):
    """
    [a b]  - [c d]  = [a-c b-d]

    Matrix + Matrix = Matrix
    """
    if shape(x) != shape(y):
        raise ShapeException
    else:
        return [x[i]-y[i] for i in range(len(x))]


def vector_sub_checks_shapes(x,y):
    """Shape rule: the vectors must be the same size."""
    vector_sub(x,y)


def vector_sum(*args):
    """vector_sum can take any number of vectors and add them together."""
    vec_sum = args[0]
    for arg in args[1:]:
        vec_sum = vector_add(vec_sum, arg)
    return vec_sum


def vector_sum_checks_shapes(*vectors):
    """Shape rule: the vectors must be the same size."""
    return vector_sum(*args)


def dot(u,v):
    """
    dot([a b], [c d])   = a * c + b * d

    dot(Vector, Vector) = Scalar
    """
    if shape(u) != shape(v):
        raise ShapeException
    else:
        return sum([ u[i]*v[j] for i, value in enumerate(v) for j,value2 in enumerate(u) if i==j])


def dot_checks_shapes(x,y):
    """Shape rule: the vectors must be the same size."""
    dot(x,y)

def vector_multiply(x,constant):
    """
    [a b]  *  Z     = [a*Z b*Z]

    Vector * Scalar = Vector
    """
    return [value*constant for value in x]

def vector_mean(*vectors):
    """
    mean([a b], [c d]) = [mean(a, c) mean(b, d)]

    mean(Vector)       = Vector
    """
    return [ element/len(vectors) for i,element in enumerate(vector_sum(*vectors))]

def magnitude(vector):
    """
    magnitude([a b])  = sqrt(a^2 + b^2)

    magnitude(Vector) = Scalar
    """
    return math.sqrt(sum([x**2 for x in vector]))

def shape_matrices(matrix):
    """shape should take a vector or matrix and return a tuple with the
    number of rows (for a vector) or the number of rows and columns
    (for a matrix.)"""
    return shape(matrix)

def matrix_row(matrix, row):
    """
           0 1  <- rows
       0 [[a b]]
       1 [[c d]]
       ^
     columns
    """
    return matrix[(row)]

def matrix_col(matrix, col):
    """
           0 1  <- rows
       0 [[a b]]
       1 [[c d]]
       ^
     columns
    """
    return [matrix[idx][col] for idx in range(len(matrix))]

def matrix_scalar_multiply(matrix, scalar):
    """
    [[a b]   *  Z   =   [[a*Z b*Z]
     [c d]]              [c*Z d*Z]]

    Matrix * Scalar = Matrix
    """
    return [[value*scalar
            for value in row]
            for row in matrix]

def matrix_vector_multiply(matrix, vector):
    """
    [[a b]   *  [x   =   [a*x+b*y
     [c d]       y]       c*x+d*y
     [e f]                e*x+f*y]

    Matrix * Vector = Vector
    """
    if shape(matrix[0]) != shape(vector):
        raise ShapeException
    else:
        multiples = [[m_value*vector[i] for i,m_value in enumerate(row)] for row in matrix]
        return [sum(line) for line in multiples]

def matrix_vector_multiply_checks_shapes(matrix, vector):
    """Shape Rule: The number of rows of the vector must equal the number of
    columns of the matrix."""
    matrix_vector_multiply(matrix, vector)


def matrix_matrix_multiply(matrix1, matrix2):
    """
    [[a b]   *  [[w x]   =   [[a*w+b*y a*x+b*z]
     [c d]       [y z]]       [c*w+d*y c*x+d*z]
     [e f]                    [e*w+f*y e*x+f*z]]

    Matrix * Matrix = Matrix
    """
    if shape(matrix1)[1] != shape(matrix2)[0]:
        raise ShapeException
    else:
        rotation = [[value[i] for value in matrix2] for i in range(len(matrix2[0]))]

        return [[dot(matrix1[i], rotation[j]) for j in range(len(rotation))] for i in range(len(matrix1))]


def matrix_matrix_multiply_checks_shapes(matrix1, matrix2):
    """Shape Rule: The number of columns of the first matrix must equal the
    number of rows of the second matrix."""
    matrix_matrix_multiply(matrix1, matrix2)


if __name__ == '__main__':
    print("Voila")
