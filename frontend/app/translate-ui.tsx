"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";

export default function TranslateUI() {
  const [sourceText, setSourceText] = useState("");
  const [translatedText, setTranslatedText] = useState("");

  const handleTranslate = async () => {
    const res = await fetch(
      `http://localhost:5000/translate?text=${sourceText}`,
    );
    if (!res.ok) throw new Error("Translation backend error");

    const { translated_text } = await res.json();
    setTranslatedText(translated_text);
  };

  return (
    <div className="max-w-2xl mx-auto p-4 space-y-6">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <h3 className="mb-3">English</h3>
          <textarea
            className="w-full h-40 p-2 border rounded-md resize-none"
            placeholder="Enter text to translate"
            value={sourceText}
            onChange={(e) => setSourceText(e.target.value)}
          />
        </div>
        <div>
          <h3 className="mb-3">Italian</h3>
          <textarea
            className="w-full h-40 p-2 border rounded-md resize-none bg-gray-100"
            placeholder="Translation will appear here"
            value={translatedText}
            readOnly
          />
        </div>
      </div>
      <Button
        onClick={handleTranslate}
        disabled={sourceText.length === 0}
        className="w-full "
      >
        Translate
      </Button>
      <p className="text-neutral-400 text-xs">
        <b className="text-black font-semibold">Note:</b> The purpose of this
        project is to learn how transformers work rather than actually building
        a good translator. I could spent an arbitrary amount of time optmising
        it, such as playing around with beam width, using more advanced
        encoding/tokenizing schemes, and using more training data but I think
        that is a waste of time.
      </p>
      <p className="text-neutral-400 text-xs">
        <b className="text-black font-semibold">Guide:</b> This translator is
        far from perfect. To help it translate better, try rephrasing your
        words. For example, instead of saying "What is your name?", you might
        try saying "What are you called?". Also, since the dataset is small,
        stick to using more general vocabulary. For example, try not to use
        names in phrases, such as "My name is Anthony". Uknown words will be
        replaced with the unknown token: [UNK].
      </p>
    </div>
  );
}
