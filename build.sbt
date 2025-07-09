val _scalaVersion = "3.6.4"

organization := "io.github.makingthematrix"
name := "aistudy"
homepage := Some(url("https://github.com/makingthematrix/aistudy"))
licenses := Seq("GPL 3.0" -> url("https://www.gnu.org/licenses/gpl-3.0.en.html"))
ThisBuild / scalaVersion := _scalaVersion
ThisBuild / versionScheme := Some("semver-spec")
Test / scalaVersion := _scalaVersion

val standardOptions = Seq(
  "-deprecation",
  "-feature",
  "-unchecked",
  "-encoding",
  "utf8"
)

val scala3Options = Seq(
  "-explain",
  "-Ysafe-init",
  "-Ycheck-all-patmat",
  "-Wunused:imports"
)

developers := List(
  Developer(
    "makingthematrix",
    "Maciej Gorywoda",
    "makingthematrix@protonmail.com",
    url("https://github.com/makingthematrix"))
)

lazy val root = (project in file("."))
  .settings(
    name := "aistudy",
    libraryDependencies ++= Seq(
      //"org.scalanlp" %% "breeze" % "2.1.0",
      //"ai.dragonfly" %% "slash" % "0.3.21",
      //"org.tensorflow" % "tensorflow-core-platform" % "1.0.0",
      "org.platanios" %% "tensorflow" % "0.4.1" classifier "darwin-cpu-x86_64",
        //Test dependencies
      "org.scalameta" %% "munit" % "1.1.0" % "test"
    ),
    scalacOptions ++= standardOptions //++ scala3Options
  )

testFrameworks += new TestFramework("munit.Framework")
