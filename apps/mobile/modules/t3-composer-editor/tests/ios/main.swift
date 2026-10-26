import UniformTypeIdentifiers

private func check(_ passed: Bool, _ message: String) {
  if !passed {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
  }
}

private let editor = ComposerTextView()
private let image = UIGraphicsImageRenderer(size: CGSize(width: 8, height: 8)).image { context in
  UIColor.red.setFill()
  context.fill(CGRect(x: 0, y: 0, width: 8, height: 8))
}
private let imageProvider = NSItemProvider(object: image)
private let pngProvider = NSItemProvider(item: image.pngData()! as NSData, typeIdentifier: UTType.png.identifier)
private let textProvider = NSItemProvider(object: "plain text" as NSString)
private let fileProvider = NSItemProvider(item: Data("file".utf8) as NSData, typeIdentifier: UTType.data.identifier)

check(editor.canPaste([imageProvider]), "An image drag must reach the composer's UITextDropDelegate")
check(editor.canPaste([pngProvider]), "A PNG representation must reach the image drop handler")
check(editor.canPaste([imageProvider, pngProvider]), "A multi-image drag must be accepted")

private let defaultEditor = UITextView()
for providers in [[textProvider], [], [fileProvider], [imageProvider, textProvider], [imageProvider, fileProvider]] {
  check(editor.canPaste(providers) == defaultEditor.canPaste(providers), "Non-image-only sessions must retain UIKit behavior")
}

editor.isReadOnly = true
check(!editor.canPaste([imageProvider]), "A read-only composer must reject image drops")
editor.isReadOnly = false
editor.isEditable = false
check(!editor.canPaste([imageProvider]), "A noneditable composer must reject image drops")
editor.isEditable = true
check(editor.canPaste([imageProvider]), "An unlocked composer must accept image drops again")
print("Composer image-drop eligibility checks passed")
