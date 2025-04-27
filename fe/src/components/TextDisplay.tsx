import React from 'react';

interface TextDisplayProps {
  text: string;
  isLoading: boolean;
}

const TextDisplay: React.FC<TextDisplayProps> = ({ text, isLoading }) => {
  return (
    <div className="w-full max-w-xl mx-auto mt-8">
      <div className="bg-gray-800 rounded-xl p-6 min-h-[150px] shadow-xl border border-gray-700">
        <h2 className="text-xl font-medium text-gray-200 mb-4 flex items-center">
          Converted Text
          {isLoading && (
            <div className="ml-3 flex items-center">
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-indigo-500 border-t-transparent"></div>
            </div>
          )}
        </h2>
        
        {isLoading ? (
          <div className="animate-pulse space-y-3">
            <div className="h-4 bg-gray-700 rounded w-3/4"></div>
            <div className="h-4 bg-gray-700 rounded"></div>
            <div className="h-4 bg-gray-700 rounded w-5/6"></div>
          </div>
        ) : text ? (
          <p className="text-gray-300 text-lg leading-relaxed">
            {text}
          </p>
        ) : (
          <p className="text-gray-500 italic text-center mt-8">
            Upload an image to see the converted text
          </p>
        )}
      </div>
    </div>
  );
};

export default TextDisplay;