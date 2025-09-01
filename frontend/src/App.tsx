import { useState, useCallback, useRef } from 'react';
import axios from 'axios';
import { Toaster, toast } from 'sonner';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Textarea } from '@/components/ui/Textarea';
import { LoaderCircle, FileText, Send, UploadCloud, Bot, User } from 'lucide-react';

const API_URL = 'http://localhost:8000/api/v1';

function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [config, setConfig] = useState({
    working_dir: './rag_storage_ollama',
    parser: 'mineru',
    host: 'http://localhost:11434',
    llm_model: 'gemma3:1b',
    vision_model: 'llava:latest',
    embedding_model: 'bge-m3:latest',
    embedding_dim: 768,
  });
  const [query, setQuery] = useState('');
  const [tableData, setTableData] = useState('');
  const [tableCaption, setTableCaption] = useState('');
  const [chatHistory, setChatHistory] = useState<{ user: string; bot: string; }[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setFiles(Array.from(event.target.files));
    }
  };

  const handleConfigChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type } = event.target;
    setConfig(prev => ({
      ...prev,
      [name]: type === 'number' ? parseInt(value, 10) : value,
    }));
  };

  const handleProcess = useCallback(async () => {
    if (files.length === 0) {
      toast.warning('No files selected.');
      return;
    }

    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    Object.entries(config).forEach(([key, value]) => {
      formData.append(key, String(value));
    });

    setIsProcessing(true);
    const promise = axios.post(`${API_URL}/process-batch`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    toast.promise(promise, {
      loading: `Processing ${files.length} file(s)...`,
      success: (response) => {
        setIsProcessing(false);
        setFiles([]);
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
        return response.data.message;
      },
      error: (error) => {
        setIsProcessing(false);
        return error.response?.data?.detail || 'An error occurred during processing.';
      },
    });
  }, [files, config]);

  const handleQuery = useCallback(async () => {
    if (!query.trim()) return;

    const multimodal_content = tableData.trim() ? [{
      type: 'table',
      table_data: tableData,
      table_caption: tableCaption,
    }] : null;

    setIsQuerying(true);
    try {
      const response = await axios.post(`${API_URL}/query`, {
        query,
        working_dir: config.working_dir,
        multimodal_content,
      });
      setChatHistory(prev => [...prev, { user: query, bot: response.data.response }]);
      setQuery('');
      setTableData('');
      setTableCaption('');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'An error occurred while querying.');
    } finally {
      setIsQuerying(false);
    }
  }, [query, config.working_dir, tableData, tableCaption]);

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100 font-sans">
      <Toaster position="top-right" richColors />
      <main className="container mx-auto p-4 md:p-8">
        <header className="text-center mb-10">
          <h1 className="text-5xl font-extrabold tracking-tight text-gray-900 dark:text-white">
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-teal-400">RAG-Anything</span> UI
          </h1>
          <p className="text-lg text-gray-500 dark:text-gray-400 mt-3 max-w-2xl mx-auto">
            Your all-in-one interface for configuring, processing, and interacting with documents using local Ollama models.
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column: Config & Upload */}
          <div className="lg:col-span-1 flex flex-col gap-8">
            <Card className="shadow-lg hover:shadow-xl transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-xl">
                  <span>⚙️</span> Configuration
                </CardTitle>
                <CardDescription>Fine-tune the parameters for your RAG pipeline.</CardDescription>
              </CardHeader>
              <CardContent className="grid grid-cols-1 gap-4">
                {Object.entries(config).map(([key, value]) => (
                  <div className="space-y-1" key={key}>
                    <label className="text-sm font-medium capitalize text-gray-700 dark:text-gray-300">
                      {key.replace(/_/g, ' ')}
                    </label>
                    <Input
                      name={key}
                      type={typeof value === 'number' ? 'number' : 'text'}
                      value={value}
                      onChange={handleConfigChange}
                      className="bg-gray-50 dark:bg-gray-800 border-gray-300 dark:border-gray-600"
                    />
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card className="shadow-lg hover:shadow-xl transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-xl">
                  <span>📤</span> Upload & Process
                </CardTitle>
                <CardDescription>Select one or more documents to ingest.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-6 text-center">
                  <UploadCloud className="mx-auto h-12 w-12 text-gray-400" />
                  <label htmlFor="file-upload" className="mt-4 text-sm font-semibold text-blue-600 hover:text-blue-500 cursor-pointer">
                    Select files to upload
                  </label>
                  <Input ref={fileInputRef} id="file-upload" type="file" multiple onChange={handleFileChange} className="sr-only" />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Supports batch uploads of various file types.</p>
                </div>
                {files.length > 0 && (
                  <div className="mt-4 space-y-2">
                    <p className="text-sm font-medium">Selected files:</p>
                    <ul className="text-sm text-gray-600 dark:text-gray-300 list-disc list-inside">
                      {files.map((file, i) => <li key={i}>{file.name}</li>)}
                    </ul>
                  </div>
                )}
                <Button onClick={handleProcess} disabled={isProcessing || files.length === 0} className="w-full mt-6 bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 text-white font-bold">
                  {isProcessing ? <LoaderCircle className="mr-2 h-4 w-4 animate-spin" /> : <FileText className="mr-2 h-4 w-4" />}
                  Process Documents
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Right Column: Chat */}
          <Card className="lg:col-span-2 shadow-lg flex flex-col h-[calc(100vh-12rem)]">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-xl">
                <span>💬</span> Chat with Your Documents
              </CardTitle>
              <CardDescription>Ask questions based on the processed content.</CardDescription>
            </CardHeader>
            <CardContent className="flex-grow flex flex-col gap-4 overflow-hidden">
              <div className="flex-grow border rounded-md p-4 overflow-y-auto bg-gray-50 dark:bg-gray-800 space-y-4">
                {chatHistory.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-gray-500 dark:text-gray-400">
                    <Bot size={48} />
                    <p className="mt-2 text-center">Chat history will appear here. <br /> Start by asking a question below.</p>
                  </div>
                ) : (
                  chatHistory.map((chat, index) => (
                    <div key={index} className="space-y-2">
                      <div className="flex gap-2 items-start justify-end">
                        <p className="bg-blue-500 text-white rounded-lg p-3 max-w-xs break-words">{chat.user}</p>
                        <User className="flex-shrink-0 h-6 w-6 text-gray-600 dark:text-gray-300 mt-1" />
                      </div>
                      <div className="flex gap-2 items-start">
                        <Bot className="flex-shrink-0 h-6 w-6 text-gray-600 dark:text-gray-300 mt-1" />
                        <p className="bg-gray-200 dark:bg-gray-700 rounded-lg p-3 max-w-xs break-words">{chat.bot}</p>
                      </div>
                    </div>
                  ))
                )}
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Multimodal Input (Optional)</label>
                <Textarea
                  placeholder="Add table data as CSV here..."
                  value={tableData}
                  onChange={(e) => setTableData(e.target.value)}
                  className="bg-gray-50 dark:bg-gray-800"
                />
                <Input
                  placeholder="Table caption..."
                  value={tableCaption}
                  onChange={(e) => setTableCaption(e.target.value)}
                  className="bg-gray-50 dark:bg-gray-800"
                />
              </div>
              <div className="flex gap-2">
                <Input
                  placeholder="Ask a question..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleQuery()}
                  disabled={isQuerying}
                  className="flex-grow bg-white dark:bg-gray-900"
                />
                <Button onClick={handleQuery} disabled={isQuerying || !query.trim()} size="icon" className="bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600">
                  {isQuerying ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}

export default App;