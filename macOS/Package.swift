// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SumApp",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "SumApp",
            targets: ["SumApp"]
        )
    ],
    dependencies: [
        // PythonKit for Python interop
        .package(url: "https://github.com/pvieito/PythonKit.git", branch: "master"),
        
        // Sparkle for auto-updates
        .package(url: "https://github.com/sparkle-project/Sparkle", from: "2.5.0"),
        
        // KeyboardShortcuts for global hotkeys
        .package(url: "https://github.com/sindresorhus/KeyboardShortcuts", from: "1.0.0"),
        
        // Defaults for user preferences
        .package(url: "https://github.com/sindresorhus/Defaults", from: "7.0.0"),
    ],
    targets: [
        .executableTarget(
            name: "SumApp",
            dependencies: [
                "PythonKit",
                "Sparkle",
                "KeyboardShortcuts",
                "Defaults"
            ],
            path: "SumApp",
            resources: [
                .process("Resources"),
                .copy("Python")
            ]
        ),
        .testTarget(
            name: "SumAppTests",
            dependencies: ["SumApp"],
            path: "SumAppTests"
        )
    ]
)