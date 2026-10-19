# Nexora ProGuard rules — keep JNI entry points and native method names
-keepclasseswithmembernames class * {
    native <methods>;
}
-keep class com.Nexora.ai.ml.NativeOps { *; }
-keepclassmembers class com.Nexora.ai.ml.NativeOps {
    public static <methods>;
}
# Keep kotlinx.serialization generated serializers
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.AnnotationsKt
-keepclassmembers class kotlinx.serialization.json.** { *** Companion; }
-keepclasseswithmembers class **$$serializer { *; }
