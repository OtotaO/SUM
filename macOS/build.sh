#!/bin/bash

# SUM for macOS - Build Script
# Builds the native Mac app with all dependencies

set -e

echo "🚀 Building SUM for macOS..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check for Xcode
if ! command -v xcodebuild &> /dev/null; then
    echo -e "${RED}❌ Xcode is required to build the Mac app${NC}"
    echo "Please install Xcode from the Mac App Store"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is required${NC}"
    echo "Please install Python 3: brew install python3"
    exit 1
fi

echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
pip3 install -r ../requirements.txt

echo -e "${BLUE}🔨 Building Swift package...${NC}"
swift build -c release

echo -e "${BLUE}📱 Creating app bundle...${NC}"

# Create app bundle structure
APP_NAME="SUM.app"
APP_PATH="build/$APP_NAME"
CONTENTS_PATH="$APP_PATH/Contents"
MACOS_PATH="$CONTENTS_PATH/MacOS"
RESOURCES_PATH="$CONTENTS_PATH/Resources"
PYTHON_PATH="$RESOURCES_PATH/Python"

# Clean and create directories
rm -rf build
mkdir -p "$MACOS_PATH"
mkdir -p "$RESOURCES_PATH"
mkdir -p "$PYTHON_PATH"

# Copy executable
cp .build/release/SumApp "$MACOS_PATH/SUM"

# Copy Python modules
echo -e "${BLUE}📋 Copying Python modules...${NC}"
cp -r ../knowledge_crystallizer.py "$PYTHON_PATH/"
cp -r ../quantum_editor_core.py "$PYTHON_PATH/"
cp -r ../graph_rag_crystallizer.py "$PYTHON_PATH/"
cp -r ../multi_agent_orchestrator.py "$PYTHON_PATH/"
cp -r ../raptor_hierarchical.py "$PYTHON_PATH/"

# Create Info.plist
echo -e "${BLUE}📝 Creating Info.plist...${NC}"
cat > "$CONTENTS_PATH/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>SUM</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>com.opensource.sum</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>SUM</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>2.0.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
    <key>NSRequiresAquaSystemAppearance</key>
    <false/>
    <key>NSHumanReadableCopyright</key>
    <string>Open Source - MIT License</string>
    <key>NSMainStoryboardFile</key>
    <string>Main</string>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>NSServices</key>
    <array>
        <dict>
            <key>NSMenuItem</key>
            <dict>
                <key>default</key>
                <string>Summarize with SUM</string>
            </dict>
            <key>NSMessage</key>
            <string>summarizeSelection</string>
            <key>NSPortName</key>
            <string>SUM</string>
            <key>NSSendTypes</key>
            <array>
                <string>NSStringPboardType</string>
            </array>
        </dict>
    </array>
</dict>
</plist>
EOF

# Create icon (placeholder)
echo -e "${BLUE}🎨 Creating app icon...${NC}"
cat > "$RESOURCES_PATH/AppIcon.iconset/icon_512x512.png" << EOF
# This would be a real icon file
# For now, using a placeholder
EOF

# Sign the app (if certificates are available)
if security find-identity -p codesigning | grep -q "Developer ID Application"; then
    echo -e "${BLUE}🔐 Signing app...${NC}"
    codesign --deep --force --verify --verbose --sign "Developer ID Application" "$APP_PATH"
else
    echo -e "${BLUE}⚠️  No signing certificate found. App will run but may show security warnings.${NC}"
fi

# Create DMG for distribution
echo -e "${BLUE}💿 Creating DMG installer...${NC}"
DMG_NAME="SUM-2.0.0.dmg"
hdiutil create -volname "SUM" -srcfolder "build/$APP_NAME" -ov -format UDZO "build/$DMG_NAME"

echo -e "${GREEN}✅ Build complete!${NC}"
echo ""
echo "📦 App bundle: build/$APP_NAME"
echo "💿 DMG installer: build/$DMG_NAME"
echo ""
echo "To run the app:"
echo "  open build/$APP_NAME"
echo ""
echo "To install:"
echo "  1. Open build/$DMG_NAME"
echo "  2. Drag SUM to Applications folder"
echo ""
echo -e "${GREEN}🎉 SUM for macOS is ready!${NC}"