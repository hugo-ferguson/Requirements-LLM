interface ComingSoonPageProps {
  title: string;
}

export function ComingSoonPage({ title }: ComingSoonPageProps) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 text-gray-500">
      <h2 className="text-xl font-semibold text-gray-700">{title}</h2>
      <p>Coming soon.</p>
    </div>
  );
}
