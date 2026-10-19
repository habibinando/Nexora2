@rem Nexora Gradle wrapper for Windows. Requires gradle\wrapper\gradle-wrapper.jar.
@rem If missing: run `gradle wrapper --gradle-version 9.8.1` once, or open in Android Studio.
@echo off
set APP_HOME=%~dp0
set CLASSPATH=%APP_HOME%gradle\wrapper\gradle-wrapper.jar
java -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*
