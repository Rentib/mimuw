package pl.edu.mimuw;

import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.ArgumentsProvider;

import java.util.stream.Stream;

import static org.junit.jupiter.params.provider.Arguments.of;
import static pl.edu.mimuw.TestMatrixData.*;

public class TestMatrixArgumentProvider implements ArgumentsProvider {

  @Override
  public Stream<? extends Arguments> provideArguments(ExtensionContext context) {
    return Stream.of(
      of(FULL_2X3),
      of(FULL_3X2),
      of(DIAGONAL_3X3),
      of(ANTI_DIAGONAL_3X3),
      of(SPARSE_2X3),
      of(SPARSE_3X2),
      of(VECTOR_2),
      of(VECTOR_3),
      of(ID_2),
      of(ID_3),
      of(ZERO_3X2)
    );
  }
}
