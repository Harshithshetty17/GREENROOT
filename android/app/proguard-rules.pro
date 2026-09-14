# The release build shrinks and obfuscates, so anything reached only by name
# has to be kept explicitly.

# The WebView calls into NativeBridge by reflection: R8 cannot see the call
# sites, and an obfuscated bridge silently stops working at runtime.
-keepclassmembers class in.greenroot.app.NativeBridge {
    @android.webkit.JavascriptInterface <methods>;
}
-keep class in.greenroot.app.NativeBridge { *; }

# Saved cards are persisted as JSON keyed by field name.
-keepclassmembers class in.greenroot.app.SavedCard { <fields>; }

# Line numbers in Play Console crash reports; the source file name is still
# hidden. Without this a stack trace from the field is unreadable.
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile
