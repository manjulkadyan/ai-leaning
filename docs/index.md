# AI Engineering, From Zero

This site is a single source of truth for going from **no AI/ML knowledge** to **professional AI engineer with deep on-device expertise**.

It is written for one specific reader: an experienced software engineer (8+ years, Android, Kotlin, mobile architecture, performance) who has *used* AI features but never learned how they work underneath. Nothing is assumed. Every concept is introduced with what it is, why it exists, an analogy, the internals, the math where it matters, runnable code, and its limits.

!!! tip "How to use this site"
    1. Start with the **Roadmap** to see the whole journey and how topics depend on each other.
    2. Follow the **Study Plan**: one lesson per day, a review every week, a milestone every month.
    3. Each lesson ends with exercises and a checklist. Don't move on until the checklist is ticked.

## What you already have that most beginners don't

Your engineering background is a real head start. Throughout the course you'll see **"Android bridge"** boxes that map new ideas to things you already know:

| You already know | You'll learn | The connection |
|---|---|---|
| Kotlin collections, `map`/`filter` | NumPy vectorization | Whole-array operations instead of loops |
| Profiling with Android Studio, systrace | Model latency and memory profiling | Same mindset, new bottlenecks |
| APK size budgets | Model quantization and compression | Shrinking an asset without breaking it |
| Coroutines and threading | Batching and async inference | Keeping the UI thread free while a model runs |
| Gradle build variants | Model variants per device tier | Ship the right artifact per hardware |

## Status

The site is being built in phases. The roadmap and study plan come first, then lessons are added one by one.
