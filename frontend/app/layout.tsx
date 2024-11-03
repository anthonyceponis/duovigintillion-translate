import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Duovigintillion Translate",
  description: "Transformer language translation implementation.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="font-mono">{children}</body>
    </html>
  );
}
