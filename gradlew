#!/bin/sh
# Nexora Gradle wrapper. Requires gradle/wrapper/gradle-wrapper.jar.
# If missing: run `gradle wrapper --gradle-version 9.8.1` once, or open in Android Studio.
APP_HOME=$(cd "$(dirname "$0")" && pwd)
CLASSPATH=$APP_HOME/gradle/wrapper/gradle-wrapper.jar
exec java -classpath "$CLASSPATH" org.gradle.wrapper.GradleWrapperMain "$@"
