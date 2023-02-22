import java.util.*;

public class Main {
  public static void main(String[] args) {
  }
}

class DoubleMatrixFactory {
  private DoubleMatrixFactory() {}
  public static IDoubleMatrix sparse(Shape shape, MatrixCellValue... values){
    return new SparseDoubleMatrix(shape, values);
  }
  public static IDoubleMatrix full(double[][] values) {
    return new FullDoubleMatrix(values);
  }
  public static IDoubleMatrix identity(int size) {
    return new IdentityDoubleMatrix(size);
  }
  public static IDoubleMatrix diagonal(double... diagonalValues) {
    return new DiagonalDoubleMatrix(diagonalValues);
  }
  public static IDoubleMatrix antiDiagonal(double... antiDiagonalValues) {
    return new AntiDiagonalDoubleMatrix(antiDiagonalValues);
  }
  public static IDoubleMatrix vector(double... values){
    double[][] xd = new double[1][values.length];
    xd[0] = values.clone();
    return new FullDoubleMatrix(xd);
  }
  public static IDoubleMatrix zero(Shape shape) {
    return new ConstDoubleMatrix(shape, 0);
  }
}

interface IDoubleMatrix {
  IDoubleMatrix times(IDoubleMatrix other);
  IDoubleMatrix times(double scalar);
  IDoubleMatrix plus(IDoubleMatrix other);
  IDoubleMatrix plus(double scalar);
  IDoubleMatrix minus(IDoubleMatrix other);
  IDoubleMatrix minus(double scalar);
  double get(int row, int column);
  double[][] data();
  double normOne();
  double normInfinity();
  double frobeniusNorm();
  String toString();
  Shape shape();
}

abstract class DoubleMatrix implements IDoubleMatrix {
  protected Shape shape;
  abstract public IDoubleMatrix times(IDoubleMatrix other);
  abstract public IDoubleMatrix times(double scalar);
  abstract public IDoubleMatrix plus(IDoubleMatrix other);
  abstract public IDoubleMatrix plus(double scalar);
  public IDoubleMatrix minus(IDoubleMatrix other) {
    return this.plus(other.times(-1));
  }
  public IDoubleMatrix minus(double scalar) { return this.plus(-scalar); }
  abstract public double get(int row, int column);
  public double[][] data() {
    double[][] xd = new double[shape.rows][shape.columns];
    for (int row = 0; row < shape.rows; row++)
      for (int column = 0; column < shape.columns; column++)
        xd[row][column] = this.get(row, column);
    return xd;
  }
  abstract public double normOne();
  abstract public double normInfinity();
  abstract public double frobeniusNorm();
  abstract public String toString();
  public Shape shape() { return shape; }
}

class IdentityDoubleMatrix extends DoubleMatrix {
  public IdentityDoubleMatrix(int size) { shape = Shape.matrix(size, size); }
  public IDoubleMatrix times(IDoubleMatrix other) {
    return FullDoubleMatrix.mult(other, this);
  }
  public IDoubleMatrix times(double scalar) {
    if (scalar == 1) return this;
    double[] xd = new double[shape.rows];
    for (int i = 0; i < xd.length; i++) xd[i] = scalar;
    return new DiagonalDoubleMatrix(xd);
  }
  public IDoubleMatrix plus(IDoubleMatrix other) {
    return FullDoubleMatrix.add(this, other);
  }
  public IDoubleMatrix plus(double scalar) {
    return FullDoubleMatrix.add(this, new ConstDoubleMatrix(shape, scalar));
  }
  public double get(int row, int column) {
    shape.assertInShape(row, column);
    return row == column ? 1 : 0;
  }
  public double normOne() { return 1; }
  public double normInfinity() { return 1; }
  public double frobeniusNorm() { return Math.sqrt(shape.rows); }
  @Override public String toString() {
    return "Type: identity\nShape: " + shape.rows +" x " + shape.columns + "\n";
  }
}

abstract class ArrayDoubleMatrix extends DoubleMatrix {
  protected double[] array;
  public IDoubleMatrix plus(double scalar) {
    return FullDoubleMatrix.add(this, new ConstDoubleMatrix(shape, scalar));
  }
  public double normOne() {
    double res = 0;
    for (var i : array) res = Math.max(res, Math.abs(i));
    return res;
  }
  public double normInfinity() { return this.normOne(); }
  public double frobeniusNorm() {
    double res = 0;
    for (var i : array) res += Math.abs(i);
    return Math.sqrt(res);
  }
}

class AntiDiagonalDoubleMatrix extends ArrayDoubleMatrix {
  public AntiDiagonalDoubleMatrix(double ...xd) {
    assert(xd.length > 0);
    shape = Shape.matrix(xd.length, xd.length);
    array = xd.clone();
  }
  public IDoubleMatrix times(IDoubleMatrix other) {
    assert(this.shape().rows == other.shape().columns);
    if (other.getClass() == IdentityDoubleMatrix.class) return this;
    if (other.getClass() == this.getClass()) {
      double[] xd = new double[array.length];
      for (int i = 0, n = xd.length; i < n; i++)
        xd[i] = this.get(i, n - i - 1) * other.get(n - i - 1, i);
      return new DiagonalDoubleMatrix(xd);
    }
    return FullDoubleMatrix.mult(this, other);
  }
  public IDoubleMatrix times(double scalar) {
    if (scalar == 0) return new ConstDoubleMatrix(shape, 0);
    if (scalar == 1) return this;
    double[] xd = array.clone();
    for (int i = 0; i < array.length; xd[i++] *= scalar);
    return new AntiDiagonalDoubleMatrix(xd);
  }
  public IDoubleMatrix plus(IDoubleMatrix other) {
    assert(shape.equals(other.shape()));
    if (this.getClass() == other.getClass()) {
      double[] xd = array.clone();
      for (int i = 0, n = xd.length; i < n; i++)
        xd[i] += other.get(i, n - i - 1);
      return new AntiDiagonalDoubleMatrix(xd);
    }
    return FullDoubleMatrix.add(this, other);
  }
  public double get(int row, int column) {
    shape.assertInShape(row, column);
    return row == shape.rows - column - 1 ? array[row] : 0;
  }
  @Override public String toString() {
    StringBuilder s = new StringBuilder("Type: antidiagonal\nShape: " + 
        shape.rows + " x " + shape.columns + "\n");
    for (int i = 0, n = array.length; i < n; i++) {
      if (i == n - 1) {
        s.append(array[i]);
        if (n == 2) s.append(" 0\n");
        if (n == 3) s.append(" 0 0\n");
        if (n >  3) s.append(" 0 ... 0\n");
      } else if (i == 0) {
        if (n == 2) s.append("0 ");
        if (n == 3) s.append("0 0 ");
        if (n >  3) s.append("0 ... 0 ");
        s.append(array[i] + "\n");
      } else {
        if (n - i - 1 == 1) s.append("0 ");
        if (n - i - 1 == 2) s.append("0 0 ");
        if (n - i - 1 >  2) s.append("0 ... 0 ");
        s.append(array[i]);
        if (i == 1) s.append(" 0");
        if (i == 2) s.append(" 0 0");
        if (i >  2) s.append(" 0 ... 0");
        s.append("\n");
      }
    }
    return s.toString();
  }
}

class DiagonalDoubleMatrix extends ArrayDoubleMatrix {
  public DiagonalDoubleMatrix(double ...xd) {
    assert(xd.length > 0);
    shape = Shape.matrix(xd.length, xd.length);
    array = xd.clone();
  }
  public IDoubleMatrix times(IDoubleMatrix other) {
    assert(this.shape().rows == other.shape().columns);
    if (other.getClass() == IdentityDoubleMatrix.class) return this;
    if (this.getClass() == other.getClass()) {
      double[] xd = array.clone();
      for (int i = 0; i < array.length; i++) xd[i] *= other.get(i, i);
      return new DiagonalDoubleMatrix(xd);
    }
    return FullDoubleMatrix.mult(this, other);
  }
  public IDoubleMatrix times(double scalar) {
    if (scalar == 0) return new ConstDoubleMatrix(shape, 0);
    if (scalar == 1) return this;
    double[] xd = array.clone();
    for (int i = 0; i < array.length; xd[i++] *= scalar);
    return new DiagonalDoubleMatrix(xd);
  }
  public IDoubleMatrix plus(IDoubleMatrix other) {
    assert(shape.equals(other.shape()));
    if (this.getClass() == other.getClass()) {
      double[] xd = array.clone();
      for (int i = 0; i < xd.length; i++) xd[i] += other.get(i, i);
      return new DiagonalDoubleMatrix(xd);
    }
    return FullDoubleMatrix.add(this, other);
  }
  public double get(int row, int column) {
    shape.assertInShape(row, column);
    return row == column ? array[row] : 0;
  }
  @Override
  public String toString() {
    StringBuilder s = new StringBuilder("Type: diagonal\nShape: " + 
        shape.rows + " x " + shape.columns + "\n");
    for (int i = 0, n = array.length; i < n; i++) {
      if (i == 0) {
        s.append(array[i]);
        if (n == 2) s.append(" 0\n");
        if (n == 3) s.append(" 0 0\n");
        if (n >  3) s.append(" 0 ... 0\n");
      } else if (i == n - 1) {
        if (n == 2) s.append("0 ");
        if (n == 3) s.append("0 0 ");
        if (n >  3) s.append("0 ... 0 ");
        s.append(array[i] + "\n");
      } else {
        if (i == 1) s.append("0 ");
        if (i == 2) s.append("0 0 ");
        if (i >  2) s.append("0 ... 0 ");
        s.append(array[i]);
        if (n - i - 1 == 1) s.append(" 0");
        if (n - i - 1 == 2) s.append(" 0 0");
        if (n - i - 1 >  2) s.append(" 0 ... 0");
        s.append("\n");
      }
    }
    return s.toString();
  }
}

class ConstDoubleMatrix extends DoubleMatrix {
  double xd;
  public ConstDoubleMatrix(Shape shape, double lol) { 
    this.shape = shape;
    this.xd    = lol; 
  }
  public IDoubleMatrix times(IDoubleMatrix other) {
    assert(this.shape().rows == other.shape().columns);
    if (other.getClass() == IdentityDoubleMatrix.class) return this;
    return xd == 0 ? new ConstDoubleMatrix(Shape.matrix(
                     shape.rows, other.shape().columns), 0)
                   : FullDoubleMatrix.mult(this, other);
  }
  public IDoubleMatrix times(double scalar) {
    return new ConstDoubleMatrix(shape, xd * scalar);
  }
  public IDoubleMatrix plus(IDoubleMatrix other) {
    assert(shape.equals(other.shape()));
    return this.getClass() == other.getClass() 
      ? new ConstDoubleMatrix(shape, this.get(0, 0) + other.get(0, 0))
      : xd == 0 ? other : other.plus(xd);
  }
  public IDoubleMatrix plus(double scalar) {
    return new ConstDoubleMatrix(shape, xd + scalar);
  }
  public double get(int row, int column) {
    shape.assertInShape(row, column); 
    return xd;
  }
  public double normOne() { return shape.rows * Math.abs(xd); }
  public double normInfinity() { return shape.columns * Math.abs(xd); }
  public double frobeniusNorm() {
    return Math.sqrt(shape.rows * shape.columns * Math.abs(xd));
  }
  @Override public String toString() {
    return "Type: const\nShape: " + shape.rows + " x " + shape.columns + "\n" 
         + "Value: " + xd + "\n";
  }
}

class FullDoubleMatrix extends DoubleMatrix {
  private double[][] v;
  public FullDoubleMatrix(double[][] xd) {
    assert(xd.length > 0 && xd[0].length > 0);
    shape = Shape.matrix(xd.length, xd[0].length);
    v = new double[shape.rows][shape.columns];
    for (int row = 0; row < shape.rows; row++)
      for (int column = 0; column < shape.columns; column++)
        v[row][column] = xd[row][column];
  }
  public static IDoubleMatrix mult(IDoubleMatrix a, IDoubleMatrix b) {
    assert(a.shape().rows == b.shape().columns);
    if (a.getClass() == IdentityDoubleMatrix.class) return b;
    if (b.getClass() == IdentityDoubleMatrix.class) return a;
    double[][] xd = new double[a.shape().rows][b.shape().columns];
    for (int row = 0; row < a.shape().rows; row++) {
      for (int column = 0; column < a.shape().columns; column++) {
        xd[row][column] = 0;
        for (int i = 0; i < b.shape().rows; i++)
          xd[row][column] += a.get(row, i) * b.get(i, column);
      }
    }
    return new FullDoubleMatrix(xd);
  }
  public static IDoubleMatrix add(IDoubleMatrix a, IDoubleMatrix b) {
    assert(a.shape().equals(b.shape()));
    if (a.getClass() == ConstDoubleMatrix.class && a.get(0, 0) == 0) return b;
    if (b.getClass() == ConstDoubleMatrix.class && b.get(0, 0) == 0) return a;
    double[][] xd = new double[a.shape().rows][a.shape().columns];
    for (int row = 0; row < a.shape().rows; row++)
      for (int column = 0; column < a.shape().columns; column++)
        xd[row][column] = a.get(row, column) + b.get(row, column);
    return new FullDoubleMatrix(xd);
  }
  public IDoubleMatrix times(IDoubleMatrix other) {
    return mult(this, other);
  }
  public IDoubleMatrix times(double scalar) {
    if (scalar == 0) return new ConstDoubleMatrix(shape, 0);
    if (scalar == 1) return this;
    double[][] xd = new double[shape.rows][shape.columns];
    for (int row = 0; row < shape.rows; row++)
      for (int column = 0; column < shape.columns; column++)
        xd[row][column] = this.get(row, column) * scalar;
    return new FullDoubleMatrix(xd);
  }
  public IDoubleMatrix plus(IDoubleMatrix other) {
    return add(this, other);
  }
  public IDoubleMatrix plus(double scalar) {
    double[][] xd = new double[shape.rows][shape.columns];
    for (int row = 0; row < shape.rows; row++)
      for (int column = 0; column < shape.columns; column++)
        xd[row][column] = this.get(row, column) + scalar;
    return new FullDoubleMatrix(xd);
  }
  public double get(int row, int column) {
    shape.assertInShape(row, column);
    return v[row][column];
  }
  public double normOne() {
    double res = 0;
    for (int column = 0, value = 0; column < shape.columns; 
         column++, res = Math.max(res, value), value = 0)
      for (int row = 0; row < shape.rows; 
           value += Math.abs(this.get(row++, column)));
    return res;
  }
  public double normInfinity() {
    double res = 0;
    for (int row = 0, value = 0; row < shape.rows;
         row++, res = Math.max(res, value), value = 0)
      for (int column = 0; column < shape.columns;
           value += Math.abs(this.get(row, column++)));
    return res;
  }
  public double frobeniusNorm() {
    double res = 0;
    for (int row = 0; row < shape.rows; row++)
      for (int column = 0; column < shape.columns;
           res += this.get(row, column) * this.get(row, column++));
    return Math.sqrt(res);
  }
  @Override public String toString() {
    StringBuilder s = new StringBuilder("Type: full\nShape: " + 
                                     shape.rows + " x " + shape.columns + "\n");
    for (int row = 0; row < shape.rows; row++, s.append("\n")) {
      for (int column = 0; column < shape.columns; column++, s.append(" ")) {
        if (0 < column && column < shape.columns - 1)
          s.append((this.get(row, column - 1) == this.get(row, column) 
                 && this.get(row, column) == this.get(row, column + 1) 
          ? "..." : this.get(row, column)));
        else s.append(this.get(row, column));
      }
    }
    return s.toString();
  }
}

class SparseDoubleMatrix extends DoubleMatrix {
  private ArrayList<MatrixCellValue> values;
  public SparseDoubleMatrix(Shape s, MatrixCellValue... xd) {
    for (var x : xd) s.assertInShape(x.row, x.column);
    shape = s;
    values = new ArrayList<MatrixCellValue>();
    values.addAll(List.of(xd));
    values.sort(new Comparator<MatrixCellValue>() {
      @Override public int compare(MatrixCellValue a, MatrixCellValue b) {
        return shape.rows * (a.row - b.row) + a.column - b.column;
      }
    });
  }
  public static SparseDoubleMatrix toSparse(IDoubleMatrix m) {
    SparseDoubleMatrix tmp = new SparseDoubleMatrix(m.shape());
    tmp.values = new ArrayList<MatrixCellValue>();
    if (m.getClass() == IdentityDoubleMatrix.class
    ||  m.getClass() == DiagonalDoubleMatrix.class) {
      for (int i = 0; i < m.shape().rows; i++)
        if (m.get(i, i) != 0)
          tmp.values.add(new MatrixCellValue(i, i, m.get(i, i)));
    } else if (m.getClass() == AntiDiagonalDoubleMatrix.class) {
      for (int i = 0, n = m.shape().rows; i < n; i++)
        if (m.get(i, n - i - 1) != 0)
          tmp.values.add(new MatrixCellValue(i, n - i - 1, m.get(i, n - i - 1)));
    } else {
      for (int i = 0; i < m.shape().rows; i++)
        for (int j = 0; j < m.shape().columns; j++)
          if (m.get(i, j) != 0)
            tmp.values.add(new MatrixCellValue(i, j, m.get(i, j)));
    }
    return tmp;
  }
  public static IDoubleMatrix mult(SparseDoubleMatrix a, SparseDoubleMatrix b) {
    return FullDoubleMatrix.mult(new FullDoubleMatrix(a.data()),
                                 new FullDoubleMatrix(b.data()));
  }
  public static IDoubleMatrix add(SparseDoubleMatrix a, SparseDoubleMatrix b) {
    return FullDoubleMatrix.add(new FullDoubleMatrix(a.data()),
                                new FullDoubleMatrix(b.data()));
  }
  public IDoubleMatrix times(IDoubleMatrix other) {
    assert(this.shape().columns == other.shape().rows);
    if (other.getClass() == IdentityDoubleMatrix.class) return this;
    if (other.getClass() == ConstDoubleMatrix.class
    &&  other.get(0, 0) == 0) return new ConstDoubleMatrix(
        Shape.matrix(shape.rows, other.shape().columns), 0);
    return mult(this, toSparse(other));
  }
  public IDoubleMatrix times(double scalar) {
    if (scalar == 0) return new ConstDoubleMatrix(shape, 0);
    if (scalar == 1) return this;
    MatrixCellValue[] m = new MatrixCellValue[values.size()];
    int idx = 0;
    for (MatrixCellValue i : values)
      m[idx++] = new MatrixCellValue(i.row, i.column, i.value * scalar);
    return new SparseDoubleMatrix(shape, m);
  }
  public IDoubleMatrix plus(IDoubleMatrix other) {
    assert(shape.equals(other.shape()));
    if (other.getClass() == ConstDoubleMatrix.class
    &&  other.get(0, 0) == 0) return this;
    return add(this, toSparse(other));
  }
  public IDoubleMatrix plus(double scalar) {
    return new FullDoubleMatrix(this.data()).plus(scalar);
  }
  public double get(int row, int column) {
    shape.assertInShape(row, column);
    for (MatrixCellValue i : values)
      if (i.row == row && i.column == column)
        return i.value;
    return 0;
  }
  public double normOne() {
    double[] x = new double[shape.rows];
    double res = 0;
    for (int i = 0; i < shape.rows; x[i++] = 0);
    for (MatrixCellValue i : values) {
      x[i.column] += Math.abs(i.value);
      res = Math.max(res, x[i.column]);
    }
    return res;
  }
  public double normInfinity() {
    double[] x = new double[shape.columns];
    double res = 0;
    for (int i = 0; i < shape.columns; x[i++] = 0);
    for (MatrixCellValue i : values) {
      x[i.row] += Math.abs(i.value);
      res = Math.max(res, x[i.row]);
    }
    return res;
  }
  public double frobeniusNorm() {
    double res = 0;
    for (MatrixCellValue i : values)
      res += i.value * i.value;
    return Math.sqrt(res);
  }
  @Override public String toString() {
    return "Type: sparse\nShape: " + shape.rows +" x " + shape.columns + "\n";
  }
}

final class Shape {
  public final int rows;
  public final int columns;
  private Shape(int rows, int columns) {
    this.rows = rows;
    this.columns = columns;
  }
  void assertInShape(int row, int column) {
    assert row >= 0;
    assert row < rows;
    assert column >= 0;
    assert column < columns;
  }
  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Shape shape = (Shape) o;
    return rows == shape.rows && columns == shape.columns;
  }
  @Override
  public int hashCode() {
    return Objects.hash(rows, columns);
  }
  public static Shape vector(int size) {
    return Shape.matrix(size, 1);
  }
  public static Shape matrix(int rows, int columns) {
    assert columns > 0;
    assert rows > 0;
    return new Shape(rows, columns);
  }
}

final class MatrixCellValue {
  public final int row;
  public final int column;
  public final double value;
  public MatrixCellValue(int row, int column, double value) {
    this.column = column;
    this.row = row;
    this.value = value;
  }
  @Override
  public String toString() {
    return "{" + value + " @[" + row + ", " + column + "]}";
  }
  public static MatrixCellValue cell(int row, int column, double value) {
    return new MatrixCellValue(row, column, value);
  }
}
