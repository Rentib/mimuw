import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar
import org.gradle.api.tasks.testing.logging.TestLogEvent.*
import org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL

plugins {
  id("com.github.johnrengelman.shadow") version "7.1.2"
  java
}

group = "pl.edu.mimuw"
version = "2022"

repositories {
  mavenCentral()
}

dependencies {
  testImplementation("org.junit.jupiter:junit-jupiter-api:5.8.2")
  testImplementation("org.junit.jupiter:junit-jupiter-params:5.8.2")
  testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine")
}

tasks.named<ShadowJar>("shadowJar") {
  mergeServiceFiles()
  manifest {
    attributes(mapOf("Main-Class" to "pl.edu.mimuw.Main"))
  }
}

tasks.test {
  enableAssertions = true
  var failedTests = false
  doFirst {
    failedTests = false
  }
  useJUnitPlatform()
  ignoreFailures = true
  testLogging {
    events = setOf(PASSED, SKIPPED, FAILED)
    exceptionFormat = FULL
  }
  afterSuite(closure<TestDescriptor, TestResult> { suite, result ->
    if (suite.parent == null) println(
      "TEST RESULTS: ${suite.displayName}\n" +
        "Passed: ${result.successfulTestCount}/${result.testCount}\t" +
        "Failed: ${result.failedTestCount}/${result.testCount}\t" +
        "Skipped: ${result.skippedTestCount}/${result.testCount}"
    )
    if (result.failedTestCount > 0) failedTests = true
  })
  maxParallelForks = Runtime.getRuntime().availableProcessors() / 2 + 1
  systemProperties["junit.jupiter.execution.parallel.enabled"] = "true"
  systemProperties["junit.jupiter.execution.parallel.mode.default"] = "concurrent"
  doLast {
    if (failedTests) throw GradleException("Some tests failed")
  }
}

fun <T, U> closure(c: (T, U) -> Unit): KotlinClosure2<T, U, Unit> = KotlinClosure2(c)
