#!/usr/bin/env sh
# Small Gradle distribution bootstrap for source archives that omit gradle-wrapper.jar.
# Version is read from gradle/wrapper/gradle-wrapper.properties so CI and local builds agree.
set -eu
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROPERTIES="$ROOT_DIR/gradle/wrapper/gradle-wrapper.properties"
if [ ! -f "$PROPERTIES" ]; then
  echo "gradlew: missing $PROPERTIES" >&2
  exit 1
fi
DIST_URL=$(sed -n 's/^distributionUrl=//p' "$PROPERTIES" | sed 's/\\:/\:/g' | tr -d '\r')
if [ -z "$DIST_URL" ]; then
  echo "gradlew: distributionUrl is missing from $PROPERTIES" >&2
  exit 1
fi
DIST_FILE=${DIST_URL##*/}
EXPECTED_SHA=$(sed -n 's/^distributionSha256Sum=//p' "$PROPERTIES" | tr -d '\r')
if [ -z "$EXPECTED_SHA" ]; then
  echo "gradlew: a pinned distributionSha256Sum is required." >&2
  exit 1
fi
DIST_NAME=${DIST_FILE%.zip}
INSTALL_NAME=${DIST_NAME%-bin}
GRADLE_USER_HOME=${GRADLE_USER_HOME:-"$HOME/.gradle"}
CACHE="$GRADLE_USER_HOME/nexora-wrapper"
INSTALL_DIR="$CACHE/$INSTALL_NAME"
GRADLE_BIN="$INSTALL_DIR/bin/gradle"

if [ ! -x "$GRADLE_BIN" ]; then
  mkdir -p "$CACHE"
  STAGE="$CACHE/.stage-$$"
  ARCHIVE="$CACHE/.${DIST_FILE}.$$"
  rm -rf "$STAGE"
  mkdir -p "$STAGE"
  echo "Gradle bootstrap: downloading $DIST_URL" >&2
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --output "$ARCHIVE" "$DIST_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$ARCHIVE" "$DIST_URL"
  else
    echo "Install curl or wget to download Gradle." >&2
    exit 1
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL_SHA=$(sha256sum "$ARCHIVE" | awk '{print $1}')
  elif command -v shasum >/dev/null 2>&1; then
    ACTUAL_SHA=$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')
  else
    echo "Install sha256sum or shasum to verify the Gradle distribution." >&2
    rm -f "$ARCHIVE"
    exit 1
  fi
  if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
    echo "Gradle ZIP checksum mismatch: expected $EXPECTED_SHA but got $ACTUAL_SHA" >&2
    rm -f "$ARCHIVE"
    exit 1
  fi
  if ! command -v unzip >/dev/null 2>&1; then
    echo "Install unzip to extract Gradle." >&2
    rm -f "$ARCHIVE"
    exit 1
  fi
  unzip -q "$ARCHIVE" -d "$STAGE"
  rm -f "$ARCHIVE"
  if [ ! -x "$STAGE/$INSTALL_NAME/bin/gradle" ]; then
    echo "Unexpected Gradle distribution layout at $STAGE/$INSTALL_NAME" >&2
    rm -rf "$STAGE"
    exit 1
  fi
  rm -rf "$INSTALL_DIR"
  mv "$STAGE/$INSTALL_NAME" "$INSTALL_DIR"
  rm -rf "$STAGE"
fi
exec "$GRADLE_BIN" -p "$ROOT_DIR" "$@"
