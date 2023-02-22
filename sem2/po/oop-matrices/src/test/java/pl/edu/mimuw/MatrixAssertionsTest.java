package pl.edu.mimuw;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ArgumentsSource;
import pl.edu.mimuw.matrix.IDoubleMatrix;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static pl.edu.mimuw.TestMatrixData.*;
import static pl.edu.mimuw.matrix.DoubleMatrixFactory.full;
import static pl.edu.mimuw.matrix.DoubleMatrixFactory.sparse;
import static pl.edu.mimuw.matrix.MatrixCellValue.cell;
import static pl.edu.mimuw.matrix.Shape.matrix;

public class MatrixAssertionsTest {

  @Test
  void testInvalidConstructThrowsOnDifferentSizes() {
    assertThrows(AssertionError.class, () -> full(new double[][]{
      new double[]{1, 2, 3},
      new double[]{1, 2, 3, 4},
    }));
  }

  @Test
  void testInvalidConstructThrowsOnOutboundIndices() {
    assertThrows(AssertionError.class, () -> sparse(matrix(3, 2),
      cell(3, 2, 1)
    ));
  }

  @Test
  void testInvalidConstructThrowsOnOutboundNegativeIndices() {
    assertThrows(AssertionError.class, () -> sparse(matrix(3, 2),
      cell(-3, -2, 1)
    ));
  }

  @Test
  void testFullInvalidConstructThrowsOnEmpty() {
    assertThrows(AssertionError.class, () -> full(new double[][]{}));
  }

  @Test
  void testFullInvalidConstructThrowsOnNull() {
    assertThrows(AssertionError.class, () -> full(null));
  }

  @ParameterizedTest
  @ArgumentsSource(TestMatrixArgumentProvider.class)
  void testGetThrowsOnNegativeRow(IDoubleMatrix m) {
    assertThrows(AssertionError.class, () -> m.get(-1, 0));
  }

  @ParameterizedTest
  @ArgumentsSource(TestMatrixArgumentProvider.class)
  void testGetThrowsOnNegativeColumn(IDoubleMatrix m) {
    assertThrows(AssertionError.class, () -> m.get(0, -1));
  }

  @ParameterizedTest
  @ArgumentsSource(TestMatrixArgumentProvider.class)
  void testFullGetThrowsTooBigRow(IDoubleMatrix m) {
    assertThrows(AssertionError.class, () -> m.get(4242, 0));
  }

  @ParameterizedTest
  @ArgumentsSource(TestMatrixArgumentProvider.class)
  void testGetThrowsOnTooBigColumn(IDoubleMatrix m) {
    assertThrows(AssertionError.class, () -> m.get(0, 4242));
  }

  @Test
  void testFullAddThrowsOnNotMatchedSizes() {
    assertThrows(AssertionError.class, () -> FULL_3X2.plus(FULL_2X3));
  }

  @Test
  void testSparseAddThrowsOnNotMatchedSizes() {
    assertThrows(AssertionError.class, () -> SPARSE_3X2.plus(SPARSE_2X3));
  }

  @Test
  void testFullTimesThrowsOnNotMatchedSizes() {
    assertThrows(AssertionError.class, () -> FULL_3X2.times(FULL_3X2));
  }

  @Test
  void testSparseTimesThrowsOnNotMatchedSizes() {
    assertThrows(AssertionError.class, () -> SPARSE_3X2.times(SPARSE_3X2));
  }
}
