/**
 * Vectorless RAG — Frontend JavaScript
 * ======================================
 * Provides API client functions, file upload, toast notifications,
 * and utility helpers for the web UI.
 */

const VectorlessRAG = {

    // ================================================================
    // API Client
    // ================================================================

    /**
     * Upload a file to the server.
     * @param {File} file - The file to upload.
     * @returns {Promise<Object>} - The server response.
     */
    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error?.message || 'Upload failed.');
        }
        return data;
    },

    /**
     * Build the hierarchical tree index for a document.
     * @param {string} docId - The document ID.
     * @returns {Promise<Object>} - The server response with tree data.
     */
    async buildIndex(docId) {
        const response = await fetch(`/api/index/${docId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });

        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error?.message || 'Index build failed.');
        }
        return data;
    },

    /**
     * Submit a query against an indexed document.
     * @param {string} query - The user's question.
     * @param {string} docId - The document ID to query.
     * @returns {Promise<Object>} - The server response with answer data.
     */
    async query(query, docId) {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                document_id: docId,
            }),
        });

        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error?.message || 'Query failed.');
        }
        return data;
    },

    /**
     * List all documents.
     * @returns {Promise<Object>} - List of documents.
     */
    async listDocuments() {
        const response = await fetch('/api/documents');
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error?.message || 'Failed to list documents.');
        }
        return data;
    },

    /**
     * Delete a document.
     * @param {string} docId - The document ID to delete.
     * @returns {Promise<Object>} - Confirmation response.
     */
    async deleteDocument(docId) {
        const response = await fetch(`/api/documents/${docId}`, {
            method: 'DELETE',
        });

        if (!data.success) {
            throw new Error(data.error?.message || 'Delete failed.');
        }
        return data;
    },

    /**
     * Build traditional vector index (chunking + NVIDIA embeddings).
     * @param {string} docId - The document ID.
     * @returns {Promise<Object>} - Server response.
     */
    async buildTraditionalIndex(docId) {
        const response = await fetch(`/api/traditional/index/${docId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });

        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error?.message || 'Traditional index build failed.');
        }
        return data;
    },

    /**
     * Submit query to Traditional RAG pipeline.
     * @param {string} query - The query question.
     * @param {string} docId - The document ID.
     * @returns {Promise<Object>} - Query result with chunks and score.
     */
    async queryTraditional(query, docId) {
        const response = await fetch('/api/traditional/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, document_id: docId }),
        });

        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error?.message || 'Traditional query failed.');
        }
        return data;
    },

    /**
     * Run side-by-side comparison between Traditional RAG and Vectorless RAG.
     * @param {string} query - The question to ask both.
     * @param {string} docId - The document ID.
     * @returns {Promise<Object>} - Comparison result with both traces.
     */
    async compare(query, docId) {
        const response = await fetch('/api/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, document_id: docId }),
        });

        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error?.message || 'Comparison failed.');
        }
        return data;
    },

    /**
     * Get indexing status for both pipelines.
     * @param {string} docId - The document ID.
     * @returns {Promise<Object>} - { traditional_indexed, vectorless_indexed }
     */
    async getIndexesStatus(docId) {
        const response = await fetch(`/api/indexes/status/${docId}`);
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error?.message || 'Failed to fetch index status.');
        }
        return data;
    },

    /**
     * Check API health.
     * @returns {Promise<Object>} - Health status.
     */
    async healthCheck() {
        const response = await fetch('/api/health');
        return await response.json();
    },

    // ================================================================
    // Toast Notifications
    // ================================================================

    /**
     * Show a toast notification.
     * @param {string} message - The message to display.
     * @param {string} type - 'success', 'error', 'warning', or 'info'.
     * @param {number} duration - Duration in ms (default 4000).
     */
    showToast(message, type = 'info', duration = 4000) {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;

        const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };
        toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${message}</span>`;

        container.appendChild(toast);

        // Auto-remove
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(40px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    },

    // ================================================================
    // Utility Functions
    // ================================================================

    /**
     * Format file size in human-readable form.
     * @param {number} bytes - File size in bytes.
     * @returns {string} - Formatted file size.
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const units = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return (bytes / Math.pow(1024, i)).toFixed(1) + ' ' + units[i];
    },

    /**
     * Simple markdown-to-HTML converter for answer display.
     * Handles: bold, italic, code, paragraphs, lists.
     * @param {string} text - Markdown text.
     * @returns {string} - HTML string.
     */
    formatMarkdown(text) {
        if (!text) return '';

        // Escape HTML
        let html = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Bold: **text**
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Italic: *text*
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // Inline code: `code`
        html = html.replace(/`(.*?)`/g, '<code>$1</code>');

        // Line breaks to paragraphs
        html = html
            .split(/\n\n+/)
            .map(para => `<p>${para.trim()}</p>`)
            .join('');

        // Single newlines within paragraphs
        html = html.replace(/\n/g, '<br>');

        return html;
    },
};

// ================================================================
// Global Initialization
// ================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Highlight active nav link
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.style.color = '#fff';
            link.style.background = 'rgba(255,255,255,0.15)';
        }
    });
});
