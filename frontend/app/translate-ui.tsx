"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ArrowRight } from "lucide-react";

export default function TranslateUI() {
  const [sourceText, setSourceText] = useState("");
  const [translatedText, setTranslatedText] = useState("");
  const [sourceLang, setSourceLang] = useState("en");
  const [targetLang, setTargetLang] = useState("es");

  const handleTranslate = () => {
    // In a real application, this is where you'd call your translation API
    setTranslatedText(`Translated: ${sourceText}`);
  };

  return (
    <div className="max-w-2xl mx-auto p-4 space-y-6">
      <div className="flex justify-center items-center space-x-4 my-6">
        <Select value={sourceLang} onValueChange={setSourceLang}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Source Language" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="en">English</SelectItem>
            <SelectItem value="es">Spanish</SelectItem>
            <SelectItem value="fr">French</SelectItem>
          </SelectContent>
        </Select>
        <ArrowRight className="mx-2" />
        <Select value={targetLang} onValueChange={setTargetLang}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Target Language" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="en">English</SelectItem>
            <SelectItem value="es">Spanish</SelectItem>
            <SelectItem value="fr">French</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <textarea
          className="w-full h-40 p-2 border rounded-md resize-none"
          placeholder="Enter text to translate"
          value={sourceText}
          onChange={(e) => setSourceText(e.target.value)}
        />
        <textarea
          className="w-full h-40 p-2 border rounded-md resize-none bg-gray-100"
          placeholder="Translation will appear here"
          value={translatedText}
          readOnly
        />
      </div>
      <Button onClick={handleTranslate} className="w-full">
        Translate
      </Button>
    </div>
  );
}
