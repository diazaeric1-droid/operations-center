import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Operator Notes — Semantic Search",
  description:
    "Retrieval-augmented search over the operator-note / downtime-event log. pgvector + local embeddings + grounded synthesis.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen font-sans antialiased">{children}</body>
    </html>
  );
}
