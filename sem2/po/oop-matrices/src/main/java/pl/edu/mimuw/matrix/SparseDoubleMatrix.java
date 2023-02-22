package pl.edu.mimuw.matrix;

import jdk.jshell.spi.SPIResolutionException;

import java.awt.event.MouseAdapter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

public class SparseDoubleMatrix extends DoubleMatrix {
    private ArrayList<MatrixCellValue> values;
    public SparseDoubleMatrix(Shape s, MatrixCellValue... xd) {
        for (var x : xd) s.assertInShape(x.row, x.column);
        shape = s;
        values = new ArrayList<MatrixCellValue>();
        values.addAll(List.of(xd));
        values.sort(new Comparator<MatrixCellValue>() {
            @Override public int compare(MatrixCellValue a, MatrixCellValue b) {
                return shape.columns * (a.row - b.row) + a.column - b.column;
            }
        });
    }
    public static SparseDoubleMatrix toSparse(IDoubleMatrix m) {
        if (m.getClass() == SparseDoubleMatrix.class) return (SparseDoubleMatrix) m;
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
        tmp.values.sort(new Comparator<MatrixCellValue>() {
            @Override public int compare(MatrixCellValue a, MatrixCellValue b) {
                return tmp.shape.columns * (a.row - b.row) + a.column - b.column;
            }
        });
        return tmp;
    }
    public static IDoubleMatrix mult(SparseDoubleMatrix a, SparseDoubleMatrix b) {
        assert(a.shape().columns == b.shape().rows);
        Shape shape = Shape.matrix(a.shape().rows, b.shape().columns);
        ArrayList<MatrixCellValue> xd  = new ArrayList<MatrixCellValue>();
        ArrayList<MatrixCellValue> res = new ArrayList<MatrixCellValue>();
        for (var i : a.values)
            for (var j : b.values)
                if (i.column == j.row)
                    xd.add(new MatrixCellValue(i.row, j.column, i.value * j.value));
        xd.sort(new Comparator<MatrixCellValue>() {
            @Override public int compare(MatrixCellValue a, MatrixCellValue b) {
                return shape.columns * (a.row - b.row) + a.column - b.column;
            }
        });
        int idx = -1;
        for (MatrixCellValue i : xd) {
            if (res.isEmpty() || res.get(idx).row != i.row || res.get(idx).column != i.column) {
                res.add(i);
                idx++;
            } else {
                double val = res.get(idx).value + i.value;
                res.set(idx, new MatrixCellValue(i.row, i.column, val));
            }
        }
        SparseDoubleMatrix sp = new SparseDoubleMatrix(shape);
        sp.values = res;
        return sp;
    }
    public static IDoubleMatrix add(SparseDoubleMatrix a, SparseDoubleMatrix b) {
        assert(a.shape().equals(b.shape()));
        ArrayList<MatrixCellValue> res = new ArrayList<MatrixCellValue>();
        ArrayList<MatrixCellValue> va = a.values, vb = b.values;
        int rows = a.shape().rows, columns = a.shape().columns;
        int na = va.size(), nb = vb.size();
        int ia = 0, ib = 0;
        while (ia < na && ib < nb) {
            int ra = va.get(ia).row, ca = va.get(ia).column;
            int rb = vb.get(ib).row, cb = vb.get(ib).column;
            double vala = va.get(ia).value, valb = vb.get(ib).value;
            if (ra == rb && ca == cb) {
                res.add(new MatrixCellValue(ra, ca, vala + valb));
                ia++; ib++;
            } else if ((ra - rb) * columns + ca - cb < 0) {
                res.add(new MatrixCellValue(ra, ca, vala));
                ia++;
            } else {
                res.add(new MatrixCellValue(rb, cb, valb));
                ib++;
            }
        }
        while (ia < na) res.add(va.get(ia++));
        while (ib < nb) res.add(vb.get(ib++));
        SparseDoubleMatrix s = new SparseDoubleMatrix(Shape.matrix(rows, columns));
        s.values = res;
        return s;
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
        StringBuilder s = new StringBuilder("Type: sparse\nShape: " + shape.rows +" x " + shape.columns + "\n");
        int row = 0, column = shape.columns - 1, dist;
        for (var i : values) {
            dist = shape.columns - 1 - column;
            if (dist > 2) s.append("0 ... 0\n");
            else if (dist == 2) s.append("0 0\n");
            else if (dist == 1) s.append("0\n");
            else if (row != 0) s.append("\n");
            while (row < i.row) {
                if (shape.columns > 2) {
                    s.append("0 ... 0\n");
                } else {
                    for (int j = 0; j < shape.columns; j++) s.append("0 ");
                    s.append("\n");
                }
                row++;
            }
            // now we are at our row
            column = 0;
            dist = i.column - column;
            if (dist > 2) s.append("0 ... 0 ");
            else if (dist == 2) s.append("0 0 ");
            else if (dist == 1) s.append("0 ");
            s.append(i.value + " ");
            column = i.column + 1;
        }
        while (row < shape.rows) { // print remaining rows
            if (shape.columns > 2) {
                s.append("0 ... 0\n");
            } else {
                for (int i = 0; i < shape.columns; i++) s.append("0 ");
                s.append("\n");
            }
            row++;
        }
        return s.toString();
    }
}