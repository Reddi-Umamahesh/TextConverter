import { useState } from 'react';

// const mockConvertImage = async (file: File): Promise<string> => {
//   await new Promise(resolve => setTimeout(resolve, 2000));
  
//   const sampleTeluguTexts = [
//     "నమస్కారం, ఎలా ఉన్నారు?",
//     "తెలుగు భాష చాలా అందమైనది మరియు సంపన్నమైనది.",
//     "ఆంధ్రప్రదేశ్ తెలుగు మాట్లాడే ప్రజల రాష్ట్రం.",
//     "భారతదేశంలో తెలుగు ఒక ప్రాచీన భాష.",
//   ];
  
//   return sampleTeluguTexts[Math.floor(Math.random() * sampleTeluguTexts.length)];
// };

export const useTeluguConverter = () => {
  const [convertedText, setConvertedText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const convertImage = async (file: File) => {
    try {
      setIsLoading(true);
      setError(null);
      const formData = new FormData();
      formData.append('image', file);

      const response = await fetch('http://localhost:3000/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        setConvertedText(data.prediction);
      } else {
        setError(data.error || "Conversion failed");
      }
      // const result = await mockConvertImage(file);
      setConvertedText(data.prediction);
    } catch (err) {
      setError('Failed to convert image. Please try again.');
      console.error('Conversion error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return {
    convertedText,
    isLoading,
    error,
    convertImage,
  };
};

export default useTeluguConverter;