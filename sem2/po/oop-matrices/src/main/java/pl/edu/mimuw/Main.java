package pl.edu.mimuw;

import pl.edu.mimuw.matrix.*;

import static pl.edu.mimuw.matrix.MatrixCellValue.cell;

public class Main {
  public static void main(String[] args) {
    Shape shape = Shape.matrix(10, 10);
    MatrixCellValue a = cell(1, 1, 1), b = cell(8, 7, 5), c = cell(2, 5, 3);
    double[][] fulldata = new double[10][10]; for (int i = 0; i < 100; i++) fulldata[i/10][i%10] = i/50+i%7;

    IDoubleMatrix sparse       = DoubleMatrixFactory.sparse(shape, a, b, c);
    IDoubleMatrix full         = DoubleMatrixFactory.full(fulldata);
    IDoubleMatrix identity     = DoubleMatrixFactory.identity(10);
    IDoubleMatrix diagonal     = DoubleMatrixFactory.diagonal(0, 0, 1, 2, 3, 6, 3, 1, 9, 4);
    IDoubleMatrix antidiagonal = DoubleMatrixFactory.antiDiagonal(0, 0, 1, 2, 3, 6, 3, 1, 9, 4);
    IDoubleMatrix vector       = DoubleMatrixFactory.vector(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
    IDoubleMatrix zero         = DoubleMatrixFactory.zero(shape);
    IDoubleMatrix constant     = DoubleMatrixFactory.constant(shape, 420.69);
    IDoubleMatrix samerows     = DoubleMatrixFactory.samerows(shape, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);

    System.out.println(sparse);
    System.out.println(full);
    System.out.println(identity);
    System.out.println(diagonal);
    System.out.println(antidiagonal);
    System.out.println(vector);
    System.out.println(zero);
    System.out.println(constant);
    System.out.println(samerows);
  }
}
