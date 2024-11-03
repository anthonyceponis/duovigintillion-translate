import TranslateUI from "@/app/translate-ui";

export default function Home() {
  return (
    <div className="w-screen h-screen flex flex-col gap-5 justify-center items-center">
      <h1 className="text-5xl font-semibold">Duovigintillion Translate</h1>
      <TranslateUI />
    </div>
  );
}
