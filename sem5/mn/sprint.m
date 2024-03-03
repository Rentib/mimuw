function lambda = sprint(A,n,v)
  [L,U,P] = lu(A);         % rozkład LUP macierzy A
  for i = 1:n              % n kroków odwrotnej metody potęgowej
    v = U \ (L \ (P * v)); % v = A^(-1) * v
    v = v / norm(v);       % v = v / ||v||
  end
  lambda = rayleigh(A,v);  % przybliżenie własnej wartości
end
