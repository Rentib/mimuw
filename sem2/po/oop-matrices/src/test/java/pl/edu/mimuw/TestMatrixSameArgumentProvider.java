package pl.edu.mimuw;

import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.ArgumentsProvider;

import java.util.stream.Stream;

import static org.junit.jupiter.params.provider.Arguments.of;
import static pl.edu.mimuw.TestMatrixData.*;

public class TestMatrixSameArgumentProvider implements ArgumentsProvider {

  @Override
  public Stream<? extends Arguments> provideArguments(ExtensionContext context) {
    return Stream.of(
      of(FULL_2X3, FULL_2X3),
      of(FULL_2X3, SPARSE_2X3),
      of(SPARSE_2X3, FULL_2X3),
      of(SPARSE_2X3, SPARSE_2X3)
    );
  }
}
