//
//  SumApp.swift
//  SUM - The Intelligent Text Platform for macOS
//
//  Created with love for the open source community
//

import SwiftUI
import AppKit

@main
struct SumApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var documentManager = DocumentManager()
    @StateObject private var intelligenceEngine = IntelligenceEngine()
    
    var body: some Scene {
        // Main document window
        DocumentGroup(newDocument: SumDocument()) { file in
            ContentView(document: file.$document)
                .environmentObject(documentManager)
                .environmentObject(intelligenceEngine)
        }
        .commands {
            CommandGroup(after: .sidebar) {
                Button("Show Intelligence Panel") {
                    NotificationCenter.default.post(
                        name: .toggleIntelligencePanel,
                        object: nil
                    )
                }
                .keyboardShortcut("i", modifiers: [.command, .shift])
            }
            
            CommandMenu("Summarize") {
                Button("Quick Summary") {
                    intelligenceEngine.summarize(density: 0.3)
                }
                .keyboardShortcut("s", modifiers: [.command, .shift])
                
                Button("Crystallize") {
                    intelligenceEngine.crystallize()
                }
                .keyboardShortcut("k", modifiers: .command)
                
                Divider()
                
                ForEach(DensityLevel.allCases, id: \.self) { level in
                    Button("\(level.description) (\(Int(level.value * 100))%)") {
                        intelligenceEngine.summarize(density: level.value)
                    }
                }
            }
        }
        
        // Settings window
        Settings {
            SettingsView()
                .environmentObject(intelligenceEngine)
        }
        
        // Menu bar app
        MenuBarExtra("SUM", systemImage: "sparkle") {
            MenuBarView()
                .environmentObject(intelligenceEngine)
        }
        .menuBarExtraStyle(.window)
    }
}

// MARK: - App Delegate

class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem?
    var pythonBridge: PythonBridge?
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        setupMenuBar()
        setupPythonBridge()
        setupServices()
        setupSpotlight()
    }
    
    private func setupMenuBar() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = statusItem?.button {
            button.image = NSImage(systemSymbolName: "sparkle", accessibilityDescription: "SUM")
        }
    }
    
    private func setupPythonBridge() {
        pythonBridge = PythonBridge()
        pythonBridge?.initialize()
    }
    
    private func setupServices() {
        NSApp.servicesProvider = ServiceProvider()
    }
    
    private func setupSpotlight() {
        // Register for Spotlight indexing
        SpotlightIndexer.shared.startIndexing()
    }
}

// MARK: - Main Content View

struct ContentView: View {
    @Binding var document: SumDocument
    @EnvironmentObject var documentManager: DocumentManager
    @EnvironmentObject var intelligenceEngine: IntelligenceEngine
    @State private var showIntelligencePanel = true
    @State private var selectedDensity: Double = 0.3
    @State private var selectedStyle: StylePersona = .neutral
    
    var body: some View {
        HSplitView {
            // Sidebar
            SidebarView()
                .frame(minWidth: 200, idealWidth: 250)
            
            // Main editor
            EditorView(text: $document.text)
                .frame(minWidth: 400)
            
            // Intelligence panel
            if showIntelligencePanel {
                IntelligencePanelView(
                    density: $selectedDensity,
                    style: $selectedStyle
                )
                .frame(minWidth: 250, idealWidth: 300)
            }
        }
        .toolbar {
            ToolbarItemGroup(placement: .navigation) {
                Button(action: toggleSidebar) {
                    Image(systemName: "sidebar.left")
                }
                
                Button(action: { showIntelligencePanel.toggle() }) {
                    Image(systemName: "sparkle")
                }
            }
            
            ToolbarItemGroup(placement: .principal) {
                Picker("Style", selection: $selectedStyle) {
                    ForEach(StylePersona.allCases, id: \.self) { style in
                        Text(style.description).tag(style)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 300)
            }
            
            ToolbarItemGroup(placement: .primaryAction) {
                Button("Summarize") {
                    Task {
                        await intelligenceEngine.summarize(
                            text: document.text,
                            density: selectedDensity,
                            style: selectedStyle
                        )
                    }
                }
                .keyboardShortcut("s", modifiers: .command)
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: .toggleIntelligencePanel)) { _ in
            showIntelligencePanel.toggle()
        }
    }
    
    private func toggleSidebar() {
        NSApp.keyWindow?.firstResponder?
            .tryToPerform(#selector(NSSplitViewController.toggleSidebar(_:)), with: nil)
    }
}

// MARK: - Editor View

struct EditorView: NSViewRepresentable {
    @Binding var text: String
    @EnvironmentObject var intelligenceEngine: IntelligenceEngine
    
    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSTextView.scrollableTextView()
        let textView = scrollView.documentView as! NSTextView
        
        textView.delegate = context.coordinator
        textView.isRichText = true
        textView.allowsUndo = true
        textView.isAutomaticQuoteSubstitutionEnabled = true
        textView.isAutomaticDashSubstitutionEnabled = true
        textView.isAutomaticTextReplacementEnabled = true
        textView.isAutomaticSpellingCorrectionEnabled = true
        textView.isContinuousSpellCheckingEnabled = true
        textView.isGrammarCheckingEnabled = true
        
        // Beautiful typography
        textView.font = NSFont.systemFont(ofSize: 14, weight: .regular)
        textView.textContainerInset = NSSize(width: 20, height: 20)
        
        return scrollView
    }
    
    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        let textView = scrollView.documentView as! NSTextView
        if textView.string != text {
            textView.string = text
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, NSTextViewDelegate {
        var parent: EditorView
        var completionWindow: CompletionWindow?
        
        init(_ parent: EditorView) {
            self.parent = parent
            super.init()
        }
        
        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            parent.text = textView.string
            
            // Trigger intelligent suggestions
            Task {
                await parent.intelligenceEngine.updateSuggestions(for: textView.string)
            }
        }
        
        func textView(_ textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
            if commandSelector == #selector(NSTextView.complete(_:)) {
                showCompletions(for: textView)
                return true
            }
            return false
        }
        
        private func showCompletions(for textView: NSTextView) {
            // Show AI-powered completions
            let location = textView.selectedRange().location
            let context = String(textView.string.prefix(location))
            
            Task {
                let suggestions = await parent.intelligenceEngine.getCompletions(for: context)
                await MainActor.run {
                    completionWindow = CompletionWindow(suggestions: suggestions)
                    completionWindow?.show(relativeTo: textView)
                }
            }
        }
    }
}

// MARK: - Intelligence Panel

struct IntelligencePanelView: View {
    @Binding var density: Double
    @Binding var style: StylePersona
    @EnvironmentObject var intelligenceEngine: IntelligenceEngine
    @State private var liveSummary = ""
    @State private var suggestions: [Suggestion] = []
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Live Summary Section
            GroupBox {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Label("Live Summary", systemImage: "doc.text")
                            .font(.headline)
                        Spacer()
                        Text("\(Int(density * 100))%")
                            .monospacedDigit()
                    }
                    
                    Slider(value: $density, in: 0.01...1.0) {
                        Text("Density")
                    }
                    .onChange(of: density) { newValue in
                        updateLiveSummary()
                    }
                    
                    ScrollView {
                        Text(liveSummary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(height: 150)
                    .background(Color(NSColor.textBackgroundColor))
                    .cornerRadius(8)
                }
            }
            
            // Suggestions Section
            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    Label("Suggestions", systemImage: "lightbulb")
                        .font(.headline)
                    
                    ScrollView {
                        VStack(alignment: .leading, spacing: 8) {
                            ForEach(suggestions) { suggestion in
                                SuggestionCard(suggestion: suggestion)
                            }
                        }
                    }
                }
            }
            
            Spacer()
            
            // Quick Actions
            GroupBox {
                VStack(spacing: 8) {
                    Button(action: exportAsPDF) {
                        Label("Export PDF", systemImage: "doc.pdf")
                            .frame(maxWidth: .infinity)
                    }
                    
                    Button(action: shareDocument) {
                        Label("Share", systemImage: "square.and.arrow.up")
                            .frame(maxWidth: .infinity)
                    }
                }
            }
        }
        .padding()
        .onAppear {
            updateLiveSummary()
            loadSuggestions()
        }
        .onReceive(intelligenceEngine.$currentSummary) { summary in
            liveSummary = summary
        }
        .onReceive(intelligenceEngine.$currentSuggestions) { newSuggestions in
            suggestions = newSuggestions
        }
    }
    
    private func updateLiveSummary() {
        Task {
            liveSummary = await intelligenceEngine.getLiveSummary(density: density)
        }
    }
    
    private func loadSuggestions() {
        Task {
            suggestions = await intelligenceEngine.getSuggestions()
        }
    }
    
    private func exportAsPDF() {
        // Export implementation
    }
    
    private func shareDocument() {
        // Share sheet implementation
    }
}

// MARK: - Models

struct SumDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.plainText, .markdown, .pdf] }
    
    var text: String
    
    init(text: String = "") {
        self.text = text
    }
    
    init(configuration: ReadConfiguration) throws {
        if let data = configuration.file.regularFileContents {
            text = String(decoding: data, as: UTF8.self)
        } else {
            text = ""
        }
    }
    
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        let data = text.data(using: .utf8) ?? Data()
        return FileWrapper(regularFileWithContents: data)
    }
}

enum DensityLevel: CaseIterable {
    case essence, tweet, elevator, executive, brief, standard, detailed, comprehensive
    
    var value: Double {
        switch self {
        case .essence: return 0.01
        case .tweet: return 0.02
        case .elevator: return 0.05
        case .executive: return 0.10
        case .brief: return 0.20
        case .standard: return 0.30
        case .detailed: return 0.50
        case .comprehensive: return 0.70
        }
    }
    
    var description: String {
        switch self {
        case .essence: return "Essence"
        case .tweet: return "Tweet"
        case .elevator: return "Elevator"
        case .executive: return "Executive"
        case .brief: return "Brief"
        case .standard: return "Standard"
        case .detailed: return "Detailed"
        case .comprehensive: return "Comprehensive"
        }
    }
}

enum StylePersona: String, CaseIterable {
    case hemingway, academic, storyteller, analyst, poet
    case executive, teacher, journalist, developer, neutral
    
    var description: String {
        rawValue.capitalized
    }
}

struct Suggestion: Identifiable {
    let id = UUID()
    let type: SuggestionType
    let text: String
    let action: () -> Void
}

enum SuggestionType {
    case continuation, improvement, connection, evidence
    
    var icon: String {
        switch self {
        case .continuation: return "arrow.right"
        case .improvement: return "sparkle"
        case .connection: return "link"
        case .evidence: return "magnifyingglass"
        }
    }
}

// MARK: - Supporting Views

struct SidebarView: View {
    var body: some View {
        List {
            Section("Library") {
                Label("All Documents", systemImage: "doc.text")
                Label("Recent", systemImage: "clock")
                Label("Favorites", systemImage: "star")
            }
            
            Section("Smart Folders") {
                Label("Summaries", systemImage: "doc.text.magnifyingglass")
                Label("Brainstorms", systemImage: "lightbulb")
                Label("Reviews", systemImage: "checkmark.circle")
            }
        }
        .listStyle(SidebarListStyle())
    }
}

struct SuggestionCard: View {
    let suggestion: Suggestion
    
    var body: some View {
        Button(action: suggestion.action) {
            HStack {
                Image(systemName: suggestion.type.icon)
                    .foregroundColor(.accentColor)
                Text(suggestion.text)
                    .multilineTextAlignment(.leading)
                Spacer()
            }
            .padding(8)
            .background(Color(NSColor.controlBackgroundColor))
            .cornerRadius(6)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Extensions

extension Notification.Name {
    static let toggleIntelligencePanel = Notification.Name("toggleIntelligencePanel")
}

// MARK: - Menu Bar View

struct MenuBarView: View {
    @EnvironmentObject var intelligenceEngine: IntelligenceEngine
    @State private var quickText = ""
    
    var body: some View {
        VStack(spacing: 12) {
            Text("Quick Capture")
                .font(.headline)
            
            TextEditor(text: $quickText)
                .frame(height: 100)
            
            HStack {
                Button("Summarize") {
                    Task {
                        await intelligenceEngine.quickSummarize(quickText)
                    }
                }
                .keyboardShortcut("s")
                
                Button("Clear") {
                    quickText = ""
                }
                .keyboardShortcut("c")
            }
            
            Divider()
            
            Button("Open Main Window") {
                NSApp.activate(ignoringOtherApps: true)
            }
            .keyboardShortcut("o")
            
            Button("Preferences...") {
                NSApp.sendAction(Selector(("showPreferencesWindow:")), to: nil, from: nil)
            }
            .keyboardShortcut(",")
            
            Divider()
            
            Button("Quit") {
                NSApp.terminate(nil)
            }
            .keyboardShortcut("q")
        }
        .padding()
        .frame(width: 300)
    }
}