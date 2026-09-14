// AGP 8.9 is the first release that compiles against API 36 (Android 16),
// which Google Play requires of new submissions from 31 August 2026.
// It in turn needs Gradle 8.11.1, pinned in gradle/wrapper.
plugins {
    id("com.android.application") version "8.9.1" apply false
    id("org.jetbrains.kotlin.android") version "2.0.21" apply false
}
