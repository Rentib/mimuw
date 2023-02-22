package pl.edu.mimuw;

import pl.edu.mimuw.matrix.IDoubleMatrix;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static pl.edu.mimuw.matrix.DoubleMatrixFactory.*;
import static pl.edu.mimuw.matrix.MatrixCellValue.cell;
import static pl.edu.mimuw.matrix.Shape.matrix;

public class TestMatrixData {

  public static final double TEST_PRECISION = 0.000001d;

  public static void assertArrayEqualsWithTestPrecision(double[][] expected, double[][] actual) {
    assertEquals(expected.length, actual.length);
    for (int i = 0; i < expected.length; i++) assertArrayEquals(expected[i], actual[i], TEST_PRECISION);
  }

  private TestMatrixData() {
  }

  public static final IDoubleMatrix FULL_2X3 = full(new double[][]{
    new double[]{1, 2, 3},
    new double[]{4, 5, 6}
  });

  public static final IDoubleMatrix FULL_3X2 = full(new double[][]{
    new double[]{1, 2},
    new double[]{3, 4},
    new double[]{5, 6}
  });

  public static final IDoubleMatrix DIAGONAL_3X3 = diagonal(7, 8, 9);

  public static final IDoubleMatrix ANTI_DIAGONAL_3X3 = antiDiagonal(10, 11, 12);

  public static final IDoubleMatrix SPARSE_2X3 = sparse(matrix(2, 3),
    cell(0, 0, 1),
    cell(0, 1, 2),
    cell(0, 2, 3),
    cell(1, 0, 4),
    cell(1, 1, 5),
    cell(1, 2, 6)
  );

  public static final IDoubleMatrix SPARSE_3X2 = sparse(matrix(3, 2),
    cell(0, 0, 1),
    cell(0, 1, 2),
    cell(1, 0, 3),
    cell(1, 1, 4),
    cell(2, 0, 5),
    cell(2, 1, 6)
  );

  public static final IDoubleMatrix VECTOR_3 = vector(15, 16, 17);

  public static final IDoubleMatrix VECTOR_2 = vector(18, 19);

  public static final IDoubleMatrix ID_2 = identity(2);

  public static final IDoubleMatrix ID_3 = identity(3);

  public static final IDoubleMatrix ZERO_3X2 = zero(matrix(3, 2));
}
